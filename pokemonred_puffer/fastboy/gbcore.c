/*
 * gbcore.c -- see gbcore.h. Port of PyBoy v2.7.0 (LGPL-3.0) DMG core.
 * Every function below names the PyBoy method it mirrors. Integer types follow the .pxd
 * declarations of the Cython build; the scheduling loop follows mb.Motherboard.tick and
 * pyboy.PyBoy._tick exactly (including breakpoint single-stepping).
 */
#include "gbcore.h"

/* Computed-goto dispatch (GCC/Clang extension); define FB_NO_THREADED to use the plain switch. */
#if defined(__GNUC__) && !defined(FB_NO_THREADED) && !defined(FB_THREADED)
#define FB_THREADED 1
#endif

#include <stdlib.h>
#include <string.h>
#ifdef GB_PROFILE
#define PROF(k) (g->prof[k]++)
#else
#define PROF(k) ((void)0)
#endif

#define INTR_VBLANK 1
#define INTR_LCDC 2
#define INTR_TIMER 4
#define INTR_SERIAL 8
#define INTR_HIGHTOLOW 16
#define OPCODE_BRK 0xDB
#define FRAME_CYCLES GB_FRAME_CYCLES
#define MAX_CYCLES GB_MAX_CYCLES

#if defined(__GNUC__)
#define HOT __attribute__((hot))
#define INLINE static inline __attribute__((always_inline))
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define HOT
#define INLINE static inline
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

static const uint64_t TIMER_DIVIDERS[4] = {10, 4, 6, 8};

static uint8_t mb_getitem_slow(GB *g, uint16_t i);
static void mb_setitem_slow(GB *g, uint16_t i, uint8_t value);
static void update_maps(GB *g);
static uint8_t mb_getitem_io_slow(GB *g, uint16_t i);
static void mb_setitem_io_slow(GB *g, uint16_t i, uint8_t value);

static void update_romx(GB *g);
static void update_rammaps(GB *g);

/* HRAM (0xFF80-0xFFFE) is plain memory in PyBoy's io-port handlers: no ticks, no bail. */
INLINE uint8_t mb_getitem_io_ports(GB *g, uint16_t i) {
    if (LIKELY(i >= 0xFF80 && i < 0xFFFF)) return g->internal_ram1[i - 0xFF80];
    return mb_getitem_io_slow(g, i);
}
INLINE void mb_setitem_io_ports(GB *g, uint16_t i, uint8_t value) {
    if (LIKELY(i >= 0xFF80 && i < 0xFFFF)) {
        g->internal_ram1[i - 0xFF80] = value;
        return;
    }
    mb_setitem_io_slow(g, i, value);
}

INLINE uint8_t mb_getitem(GB *g, uint16_t i) {
    if (i < 0x4000) {
        if (UNLIKELY(g->bootrom_enabled && i <= 0xFF)) return g->bootrom[i];
        return g->rom0[i];
    }
    if (i < 0x8000) return g->romx[i - 0x4000];
    const uint8_t *p = g->rmap[i >> 8];
    if (LIKELY(p != NULL)) return p[i & 0xFF];
    if (i >= 0xFF00) return mb_getitem_io_ports(g, i);
    return mb_getitem_slow(g, i);
}

INLINE void mb_setitem(GB *g, uint16_t i, uint8_t value) {
    uint8_t *p = g->wmap[i >> 8];
    if (LIKELY(p != NULL)) {
        p[i & 0xFF] = value;
        return;
    }
    if (i >= 0xFF00) {
        mb_setitem_io_ports(g, i, value);
        return;
    }
    mb_setitem_slow(g, i, value);
}



/* ===================================================================== Timer (timer.py) */
static void timer_reset(GB *g) {
    g->DIV_counter = 0;
    g->TIMA_counter = 0;
    g->DIV = 0;
}

INLINE int timer_tick(GB *g, uint64_t _cycles) {
    uint64_t cycles = _cycles - g->timer_last_cycles;
    if (cycles == 0) return 0;
    g->timer_last_cycles = _cycles;
    g->DIV_counter += cycles;
    g->DIV += g->DIV_counter >> 8;
    g->DIV_counter &= 0xFF;
    g->DIV &= 0xFF;
    if ((g->TAC & 4) == 0) {
        g->timer_cti = MAX_CYCLES;
        return 0;
    }
    g->TIMA_counter += cycles;
    uint8_t divider = (uint8_t)TIMER_DIVIDERS[g->TAC & 3];
    int ret = 0;
    while (g->TIMA_counter >= (uint64_t)(1 << divider)) {
        g->TIMA_counter -= (uint64_t)(1 << divider);
        g->TIMA += 1;
        if (g->TIMA > 0xFF) {
            g->TIMA = g->TMA;
            g->TIMA &= 0xFF;
            ret = 1;
        }
    }
    g->timer_cti = (int64_t)((((uint64_t)0x100 - g->TIMA) << divider) - g->TIMA_counter);
    return ret;
}

/* ===================================================================== Serial (serial.py) */
INLINE int serial_tick(GB *g, uint64_t _cycles) {
    uint64_t cycles = _cycles - g->serial_last_cycles;
    if (cycles == 0) return 0;
    g->serial_last_cycles = _cycles;
    g->serial_clock += cycles;
    int interrupt = 0;
    if (g->transfer_enabled && g->serial_clock >= g->serial_clock_target) {
        g->SC &= 0x80;
        g->transfer_enabled = 0;
        g->serial_clock_target = MAX_CYCLES;
        interrupt = 1;
    }
    g->serial_cti = (int64_t)(g->serial_clock_target - g->serial_clock);
    return interrupt;
}

static void serial_set_SC(GB *g, uint8_t value) {
    g->SC = value;
    g->transfer_enabled = (int)(g->SC & 0x80);
    g->internal_clock = (int)(g->SC & 1);
    if (g->internal_clock) {
        g->serial_clock_target = g->serial_clock + 8 * 128;
    } else {
        g->transfer_enabled = 0;
        g->serial_clock_target = MAX_CYCLES;
    }
    g->serial_cti = (int64_t)(g->serial_clock_target - g->serial_clock);
}

/* ===================================================================== Sound (sound.py, emulate=False) */
INLINE void sound_tick(GB *g, uint64_t _cycles) {
    g->sound_last_cycles = _cycles; /* emulate=False: only bookkeeping */
}

static void sound_reset_apu_div(GB *g) {
    g->sound_cycles_target_512Hz = (double)MAX_CYCLES; /* emulate=False branch */
}

/* ===================================================================== Interaction (interaction.py) */
static int interaction_key_event(GB *g, int key) {
    uint8_t _d = g->directional, _s = g->standard;
    switch (key) {
    case 3: g->directional = g->directional & ~(1 << 0); break; /* PRESS_ARROW_RIGHT */
    case 4: g->directional = g->directional & ~(1 << 1); break; /* PRESS_ARROW_LEFT */
    case 1: g->directional = g->directional & ~(1 << 2); break; /* PRESS_ARROW_UP */
    case 2: g->directional = g->directional & ~(1 << 3); break; /* PRESS_ARROW_DOWN */
    case 5: g->standard = g->standard & ~(1 << 0); break;       /* PRESS_BUTTON_A */
    case 6: g->standard = g->standard & ~(1 << 1); break;       /* PRESS_BUTTON_B */
    case 7: g->standard = g->standard & ~(1 << 2); break;       /* PRESS_BUTTON_SELECT */
    case 8: g->standard = g->standard & ~(1 << 3); break;       /* PRESS_BUTTON_START */
    case 11: g->directional = g->directional | (1 << 0); break; /* RELEASE_ARROW_RIGHT */
    case 12: g->directional = g->directional | (1 << 1); break; /* RELEASE_ARROW_LEFT */
    case 9: g->directional = g->directional | (1 << 2); break;  /* RELEASE_ARROW_UP */
    case 10: g->directional = g->directional | (1 << 3); break; /* RELEASE_ARROW_DOWN */
    case 13: g->standard = g->standard | (1 << 0); break;       /* RELEASE_BUTTON_A */
    case 14: g->standard = g->standard | (1 << 1); break;       /* RELEASE_BUTTON_B */
    case 15: g->standard = g->standard | (1 << 2); break;       /* RELEASE_BUTTON_SELECT */
    case 16: g->standard = g->standard | (1 << 3); break;       /* RELEASE_BUTTON_START */
    default: break;
    }
    return (((_d ^ g->directional) & _d) != 0) || (((_s ^ g->standard) & _s) != 0);
}

static uint8_t interaction_pull(GB *g, uint8_t joystickbyte) {
    int P14 = (joystickbyte >> 4) & 1;
    int P15 = (joystickbyte >> 5) & 1;
    int joystickByte = 0xFF & (joystickbyte | 0xCF);
    if (P14 && P15) {
        joystickByte = 0xF;
    } else if (!P14 && !P15) {
        /* pass */
    } else if (!P14) {
        joystickByte &= g->directional;
    } else if (!P15) {
        joystickByte &= g->standard;
    }
    return (uint8_t)(joystickByte | 0xC0);
}

/* ===================================================================== LCD registers (lcd.py) */
static void lcdc_set(GB *g, uint64_t value) {
    g->lcdc_value = (uint8_t)value;
    g->lcd_enable = (int)(value & (1 << 7));
    g->windowmap_select = (int)(value & (1 << 6));
    g->window_enable = (int)(value & (1 << 5));
    g->tiledata_select = (int)(value & (1 << 4));
    g->backgroundmap_select = (int)(value & (1 << 3));
    g->sprite_height = (int)(value & (1 << 2));
    g->sprite_enable = (int)(value & (1 << 1));
    g->background_enable = (int)(value & (1 << 0));
    g->backgroundmap_offset = g->backgroundmap_select == 0 ? 0x1800 : 0x1C00;
    g->windowmap_offset = g->windowmap_select == 0 ? 0x1800 : 0x1C00;
}

INLINE uint8_t stat_set_mode(GB *g, uint8_t mode) {
    if (g->stat_mode == mode) return 0;
    g->stat_mode = mode;
    g->stat_value &= 0xFC;
    g->stat_value |= mode;
    if (mode != 3 && (g->stat_value & (1 << (mode + 3)))) return INTR_LCDC;
    return 0;
}

INLINE uint8_t stat_update_LYC(GB *g, uint8_t LYC, uint8_t LY) {
    if (LYC == LY) {
        g->stat_value |= 4;
        if (g->stat_value & 0x40) return INTR_LCDC;
    } else {
        g->stat_value &= 0xFB;
    }
    return 0;
}

static void stat_set(GB *g, uint64_t value) {
    value &= 0x78;
    g->stat_value &= 0x87;
    g->stat_value |= (uint8_t)value;
}

static int palette_set(GBPalette *p, uint64_t value) {
    if ((uint64_t)p->value == value) return 0;
    p->value = (uint8_t)value;
    for (int x = 0; x < 4; x++) p->lookup[x] = p->palette_mem_rgb[(value >> x * 2) & 3];
    return 1;
}

static void palette_init(GBPalette *p, uint8_t value, const uint32_t *rgb) {
    p->value = 0;
    for (int i = 0; i < 4; i++) {
        p->lookup[i] = 0;
        p->palette_mem_rgb[i] = rgb[i];
    }
    palette_set(p, value);
}

static void lcd_set_lcdc(GB *g, uint8_t value) {
    int _lcd_enable = g->lcd_enable;
    lcdc_set(g, value);
    if (_lcd_enable && !g->lcd_enable) {
        g->clock = 0;
        g->clock_target = 0;
        g->lcd_ctf = 0;
        stat_set_mode(g, 0);
        g->LY = 0;
    } else if (!_lcd_enable && g->lcd_enable) {
        g->clock_target = 0;
        g->first_frame = 1;
        g->reset = 1;
    }
}

/* ===================================================================== Renderer (lcd.Renderer, DMG) */
/* The tile/sprite caches in PyBoy are a pure function of VRAM (invalidated on every VRAM write
 * through the memory bus), so we decode 2bpp rows directly. */
INLINE uint8_t tile_px(const GB *g, int row, int xx) {
    uint8_t b1 = g->VRAM0[row * 2], b2 = g->VRAM0[row * 2 + 1];
    return (uint8_t)((((b2 >> (7 - xx)) & 1) << 1) | ((b1 >> (7 - xx)) & 1));
}

INLINE void get_tile(const GB *g, uint8_t y, uint8_t x, uint16_t offset, int *tile_out, int *yy_out) {
    uint64_t tile_addr = offset + ((y / 8) * 32) % 0x400 + (x / 8) % 32;
    int tile = g->VRAM0[tile_addr];
    if (!g->tiledata_select) tile = (tile ^ 0x80) + 128;
    *tile_out = tile;
    *yy_out = 8 * tile + y % 8;
}

INLINE void put_pixel(GB *g, uint32_t pixel, int x, int y, int xx, int yy, uint32_t bg_priority_apply) {
    uint8_t col0 = (tile_px(g, yy, xx) == 0) & 1;
    g->screenbuffer[y * GB_COLS + x] = pixel;
    g->screenbuffer_attributes[y * GB_COLS + x] = (uint8_t)(bg_priority_apply | col0);
}

static int scanline_background(GB *g, int y, int _x, int bx, int by, int cols) {
    int b_yy = 0, bt = 0;
    uint8_t b1 = 0, b2 = 0;
    uint32_t *out = g->screenbuffer + y * GB_COLS;
    uint8_t *attr = g->screenbuffer_attributes + y * GB_COLS;
    for (int x = _x; x < _x + cols; x++) {
        int b_xx = (x + (bx & 7)) % 8;
        if (b_xx == 0 || x == 0) {
            get_tile(g, (uint8_t)(y + by), (uint8_t)(x + bx), g->backgroundmap_offset, &bt, &b_yy);
            b1 = g->VRAM0[b_yy * 2];
            b2 = g->VRAM0[b_yy * 2 + 1];
        }
        int sh = 7 - b_xx;
        uint8_t c = (uint8_t)((((b2 >> sh) & 1) << 1) | ((b1 >> sh) & 1));
        out[x] = g->BGP.lookup[c];
        attr[x] = (uint8_t)(c == 0);
    }
    return cols;
}

static int scanline_window(GB *g, int y, int _x, int wx, int wy, int cols) {
    (void)wy;
    int yy = 0, wt = 0;
    uint8_t b1 = 0, b2 = 0;
    uint32_t *out = g->screenbuffer + y * GB_COLS;
    uint8_t *attr = g->screenbuffer_attributes + y * GB_COLS;
    for (int x = _x; x < _x + cols; x++) {
        int xx = (x - wx) % 8;
        if (xx == 0 || x == _x) {
            get_tile(g, (uint8_t)g->ly_window, (uint8_t)(x - wx), g->windowmap_offset, &wt, &yy);
            b1 = g->VRAM0[yy * 2];
            b2 = g->VRAM0[yy * 2 + 1];
        }
        int sh = 7 - xx;
        uint8_t c = (uint8_t)((((b2 >> sh) & 1) << 1) | ((b1 >> sh) & 1));
        out[x] = g->BGP.lookup[c];
        attr[x] = (uint8_t)(c == 0);
    }
    return cols;
}

static int scanline_blank(GB *g, int y, int _x, int cols) {
    for (int x = _x; x < _x + cols; x++) {
        g->screenbuffer[y * GB_COLS + x] = g->BGP.lookup[0];
        g->screenbuffer_attributes[y * GB_COLS + x] = 0;
    }
    return cols;
}

static void renderer_scanline(GB *g, int y) {
    if (g->disable_renderer) return;
    int bx = g->SCX, by = g->SCY;
    int wx = g->WX - 7, wy = g->WY;
    int x = 0;
    if (g->window_enable && wy <= y && wx < GB_COLS) {
        g->ly_window += 1;
        if (wx > x) x += scanline_background(g, y, x, bx, by, wx);
        scanline_window(g, y, x, wx, wy, GB_COLS - x);
    } else if (g->background_enable) {
        scanline_background(g, y, x, bx, by, GB_COLS);
    } else {
        scanline_blank(g, y, x, GB_COLS);
    }
    if (y == 143) g->ly_window = -1;
}

static void sort_sprites(GB *g, int sprite_count) {
    for (int i = 1; i < sprite_count; i++) {
        int key = g->sprites_to_render[i];
        int j = i - 1;
        while (j >= 0 && key > g->sprites_to_render[j]) {
            g->sprites_to_render[j + 1] = g->sprites_to_render[j];
            j -= 1;
        }
        g->sprites_to_render[j + 1] = key;
    }
}

static void renderer_scanline_sprites(GB *g, int ly) {
    if (!g->sprite_enable || g->disable_renderer) return;
    int spriteheight = g->sprite_height ? 16 : 8;
    int sprite_count = 0;
    for (int n = 0; n < 0xA0; n += 4) {
        int y = g->OAM[n] - 16;
        int x = g->OAM[n + 1] - 8;
        if (y <= ly && ly < y + spriteheight) {
            g->sprites_to_render[sprite_count] = (int)((uint32_t)x << 16) | n;
            sprite_count += 1;
        }
        if (sprite_count == 10) break;
    }
    sort_sprites(g, sprite_count);
    for (int s = 0; s < sprite_count; s++) {
        int n = g->sprites_to_render[s] & 0xFF;
        int y = g->OAM[n] - 16;
        int x = g->OAM[n + 1] - 8;
        int tileindex = g->OAM[n + 2];
        if (spriteheight == 16) tileindex &= 0xFE;
        int attributes = g->OAM[n + 3];
        int xflip = attributes & 0x20;
        int yflip = attributes & 0x40;
        int spritepriority = (attributes & 0x80) != 0; /* ignore_priority is False */
        GBPalette *pal = (attributes & 0x10) ? &g->OBP1 : &g->OBP0;
        int dy = ly - y;
        int yy = yflip ? spriteheight - dy - 1 : dy;
        for (int dx = 0; dx < 8; dx++) {
            int xx = xflip ? 7 - dx : dx;
            uint8_t color_code = tile_px(g, 8 * tileindex + yy, xx);
            if (0 <= x && x < GB_COLS && !(color_code == 0)) {
                uint32_t pixel = pal->lookup[color_code];
                if (spritepriority) {
                    if (g->screenbuffer_attributes[ly * GB_COLS + x] & 1) g->screenbuffer[ly * GB_COLS + x] = pixel;
                } else {
                    g->screenbuffer[ly * GB_COLS + x] = pixel;
                }
            }
            x += 1;
        }
    }
}

static void renderer_blank_screen(GB *g) {
    for (int i = 0; i < GB_ROWS * GB_COLS; i++) {
        g->screenbuffer[i] = g->BGP.lookup[0];
        g->screenbuffer_attributes[i] = 0;
    }
}

/* ===================================================================== LCD tick (lcd.LCD.tick) */
INLINE uint8_t lcd_tick(GB *g, uint64_t _cycles) {
    uint64_t cycles = _cycles - g->lcd_last_cycles;
    if (cycles == 0) return 0;
    g->lcd_last_cycles = _cycles;
    uint8_t interrupt_flag = 0;
    g->clock += cycles >> g->speed_shift;

    if (g->clock >= g->clock_target) {
        if (g->lcd_enable && (g->LY == 153 || g->reset)) {
            if (g->reset) {
                g->clock = 0;
                g->clock_target = 0;
                stat_set_mode(g, 0);
                g->reset = 0;
            }
            g->frame_done = 1;
            g->LY = 0;
            g->clock %= FRAME_CYCLES;
            g->clock_target = 0;
            g->next_stat_mode = 2;
            interrupt_flag |= stat_set_mode(g, g->next_stat_mode);
            g->clock_target += 80;
            g->next_stat_mode = 3;
            interrupt_flag |= stat_update_LYC(g, g->LYC, g->LY);
        } else if (g->lcd_enable) {
            interrupt_flag |= stat_set_mode(g, g->next_stat_mode);
            if (g->stat_mode == 2) {
                g->LY += 1;
                g->clock_target += 80;
                g->next_stat_mode = 3;
                interrupt_flag |= stat_update_LYC(g, g->LYC, g->LY);
            } else if (g->stat_mode == 3) {
                g->clock_target += 170;
                g->next_stat_mode = 0;
            } else if (g->stat_mode == 0) {
                g->clock_target += 206;
                int bx = g->SCX, by = g->SCY, wx = g->WX - 7, wy = g->WY;
                g->scanlineparameters[g->LY][0] = (uint8_t)bx;
                g->scanlineparameters[g->LY][1] = (uint8_t)by;
                g->scanlineparameters[g->LY][2] = (uint8_t)(wx + 7);
                g->scanlineparameters[g->LY][3] = (uint8_t)wy;
                g->scanlineparameters[g->LY][4] = (uint8_t)g->tiledata_select;
                renderer_scanline(g, g->LY);
                renderer_scanline_sprites(g, g->LY);
                if (g->LY < 143) g->next_stat_mode = 2;
                else g->next_stat_mode = 1;
            } else if (g->stat_mode == 1) {
                g->clock_target += 456;
                g->next_stat_mode = 1;
                g->LY += 1;
                interrupt_flag |= stat_update_LYC(g, g->LYC, g->LY);
                if (g->LY == 144) {
                    interrupt_flag |= INTR_VBLANK;
                    if (g->first_frame) {
                        renderer_blank_screen(g);
                        g->first_frame = 0;
                    }
                }
            }
        } else {
            g->frame_done = 1;
            g->clock %= FRAME_CYCLES;
            g->clock_target = FRAME_CYCLES;
            renderer_blank_screen(g);
        }
    }
    g->lcd_cti = (int64_t)((g->clock_target - g->clock) << g->speed_shift);
    g->lcd_ctf = (int64_t)(((uint64_t)FRAME_CYCLES - g->clock) << g->speed_shift);
    return interrupt_flag;
}

/* ===================================================================== Memory bus (mb.Motherboard) */
static void mb_transfer_DMA(GB *g, uint8_t src) {
    int dst = 0xFE00;
    int offset = src * 0x100;
    for (int n = 0; n < 0xA0; n++) mb_setitem(g, (uint16_t)(dst + n), mb_getitem(g, (uint16_t)(n + offset)));
}

static uint8_t mb_getitem_io_slow(GB *g, uint16_t i) {
    PROF(7);
    if (0xFF00 <= i && i < 0xFF4C) {
        if (0xFF01 <= i && i <= 0xFF02) {
            if (serial_tick(g, (uint64_t)g->cycles)) g->interrupts_flag_register |= INTR_SERIAL;
            if (i == 0xFF01) return (uint8_t)g->SB;
            else return (uint8_t)g->SC;
        } else if (0xFF04 <= i && i <= 0xFF07) {
            if (timer_tick(g, (uint64_t)g->cycles)) g->interrupts_flag_register |= INTR_TIMER;
            if (i == 0xFF04) return (uint8_t)g->DIV;
            else if (i == 0xFF05) return (uint8_t)g->TIMA;
            else if (i == 0xFF06) return (uint8_t)g->TMA;
            else return (uint8_t)g->TAC;
        } else if (i == 0xFF0F) {
            return g->interrupts_flag_register;
        } else if (0xFF10 <= i && i < 0xFF40) {
            sound_tick(g, (uint64_t)g->cycles);
            return 0; /* sound.get with emulate=False */
        } else if (0xFF40 <= i && i <= 0xFF4B) {
            uint8_t li = lcd_tick(g, (uint64_t)g->cycles);
            if (li) g->interrupts_flag_register |= li;
            switch (i) {
            case 0xFF40: return g->lcdc_value;
            case 0xFF41: return g->stat_value;
            case 0xFF42: return g->SCY;
            case 0xFF43: return g->SCX;
            case 0xFF44: return g->LY;
            case 0xFF45: return g->LYC;
            case 0xFF46: return 0x00;
            case 0xFF47: return g->BGP.value;
            case 0xFF48: return g->OBP0.value;
            case 0xFF49: return g->OBP1.value;
            case 0xFF4A: return g->WY;
            default: return g->WX;
            }
        } else {
            return g->io_ports[i - 0xFF00];
        }
    } else if (0xFF4C <= i && i < 0xFF80) {
        return g->non_io_internal_ram1[i - 0xFF4C];
    } else if (0xFF80 <= i && i < 0xFFFF) {
        return g->internal_ram1[i - 0xFF80];
    } else {
        return g->interrupts_enabled_register;
    }
}

static uint8_t mb_getitem_slow(GB *g, uint16_t i) {
    PROF(3);
    if (i < 0x4000) {
        if (g->bootrom_enabled && i <= 0xFF) return g->bootrom[i];
        return g->rombanks[(int64_t)g->rombank_selected_low * 0x4000 + i];
    } else if (i < 0x8000) {
        return g->rombanks[(int64_t)g->rombank_selected * 0x4000 + (i - 0x4000)];
    } else if (i < 0xA000) {
        return g->VRAM0[i - 0x8000];
    } else if (i < 0xC000) {
        if (!g->rambank_enabled) return 0xFF;
        return g->rambanks[(int64_t)g->rambank_selected * 8192 + (i - 0xA000)];
    } else if (i < 0xE000) {
        return g->internal_ram0[i - 0xC000];
    } else if (i < 0xFE00) {
        return mb_getitem_slow(g, (uint16_t)(i - 0x2000));
    } else if (i < 0xFEA0) {
        return g->OAM[i - 0xFE00];
    } else if (i < 0xFF00) {
        return g->non_io_internal_ram0[i - 0xFEA0];
    } else {
        return mb_getitem_io_ports(g, i);
    }
}

static void mbc3_setitem(GB *g, uint16_t address, uint8_t value) {
    if (address < 0x2000) {
        if ((value & 0x0F) == 0x0A) g->rambank_enabled = 1;
        else g->rambank_enabled = 0;
        update_rammaps(g);
    } else if (address < 0x4000) {
        value &= 0x7F;
        if (value == 0) value = 1;
        g->rombank_selected = (uint16_t)(value % g->external_rom_count);
        update_romx(g);
    } else if (address < 0x6000) {
        if (0x08 <= value && value <= 0x0C) g->rambank_selected = value;
        else g->rambank_selected = (uint16_t)(value % g->external_ram_count);
        update_rammaps(g);
    } else if (address < 0x8000) {
        /* RTC latch: no RTC on this cartridge */
    } else if (0xA000 <= address && address < 0xC000) {
        if (g->rambank_enabled) {
            if (g->rambank_selected <= 0x03) g->rambanks[(int64_t)g->rambank_selected * 8192 + (address - 0xA000)] = value;
        }
    }
}

static void mb_setitem_io_slow(GB *g, uint16_t i, uint8_t value) {
    PROF(8);
    if (0xFF00 <= i && i < 0xFF4C) {
        if (i == 0xFF00) {
            g->io_ports[0] = interaction_pull(g, value);
        } else if (0xFF01 <= i && i <= 0xFF02) {
            if (serial_tick(g, (uint64_t)g->cycles)) g->interrupts_flag_register |= INTR_SERIAL;
            if (i == 0xFF01) {
                g->serialbuffer_count += 1;
                g->serialbuffer_count &= 0x3FF;
                g->SB = 0xFF; /* serial.set_SB */
            } else {
                serial_set_SC(g, value);
            }
        } else if (0xFF04 <= i && i <= 0xFF07) {
            if (timer_tick(g, (uint64_t)g->cycles)) g->interrupts_flag_register |= INTR_TIMER;
            if (i == 0xFF04) {
                if (g->DIV & ((uint64_t)0x10 << g->sound_speed_shift)) {
                    sound_tick(g, (uint64_t)g->cycles);
                    sound_reset_apu_div(g);
                }
                timer_reset(g);
            } else if (i == 0xFF05) {
                g->TIMA = value;
            } else if (i == 0xFF06) {
                g->TMA = value;
            } else {
                g->TAC = value & 7;
            }
        } else if (i == 0xFF0F) {
            g->interrupts_flag_register = value;
        } else if (0xFF10 <= i && i < 0xFF40) {
            sound_tick(g, (uint64_t)g->cycles); /* sound.set: emulate=False -> no-op */
        } else if (0xFF40 <= i && i <= 0xFF4B) {
            uint8_t li = lcd_tick(g, (uint64_t)g->cycles);
            if (li) g->interrupts_flag_register |= li;
            switch (i) {
            case 0xFF40: lcd_set_lcdc(g, value); break;
            case 0xFF41: stat_set(g, value); break;
            case 0xFF42: g->SCY = value; break;
            case 0xFF43: g->SCX = value; break;
            case 0xFF44: return; /* read-only; note: returns before cpu.bail */
            case 0xFF45: g->LYC = value; break;
            case 0xFF46: mb_transfer_DMA(g, value); break;
            case 0xFF47: palette_set(&g->BGP, value); break;
            case 0xFF48: palette_set(&g->OBP0, value); break;
            case 0xFF49: palette_set(&g->OBP1, value); break;
            case 0xFF4A: g->WY = value; break;
            default: g->WX = value; break;
            }
        } else {
            g->io_ports[i - 0xFF00] = value;
        }
        g->bail = 1;
    } else if (0xFF4C <= i && i < 0xFF80) {
        if (g->bootrom_enabled && i == 0xFF50 && (value == 0x1 || value == 0x11)) {
            g->bootrom_enabled = 0;
            g->bail = 1;
        } else {
            g->non_io_internal_ram1[i - 0xFF4C] = value;
        }
    } else if (0xFF80 <= i && i < 0xFFFF) {
        g->internal_ram1[i - 0xFF80] = value;
    } else {
        g->interrupts_enabled_register = value;
        g->bail = 1;
    }
}

static void mb_setitem_slow(GB *g, uint16_t i, uint8_t value) {
    PROF(4);
    if (i < 0x8000) {
        mbc3_setitem(g, i, value);
        g->bail = 1;
    } else if (i < 0xA000) {
        g->VRAM0[i - 0x8000] = value;
    } else if (i < 0xC000) {
        mbc3_setitem(g, i, value);
    } else if (i < 0xE000) {
        g->internal_ram0[i - 0xC000] = value;
    } else if (i < 0xFE00) {
        mb_setitem_slow(g, (uint16_t)(i - 0x2000), value);
    } else if (i < 0xFEA0) {
        g->OAM[i - 0xFE00] = value;
    } else if (i < 0xFF00) {
        g->non_io_internal_ram0[i - 0xFEA0] = value;
    } else {
        mb_setitem_io_ports(g, i, value);
    }
}

/* Page tables mirror mb_getitem_slow / mb_setitem_slow for regions that are plain memory.
 * ROM (0x0000-0x7FFF) is resolved via rom0/romx; the tables cover 0x8000-0xFDFF. */
static void update_romx(GB *g) {
    PROF(5);
    g->rom0 = g->rombanks + (int64_t)g->rombank_selected_low * 0x4000;
    g->romx = g->rombanks + (int64_t)g->rombank_selected * 0x4000;
}

static void update_rammaps(GB *g) {
    for (int p = 0xA0; p < 0xC0; p++) {
        g->rmap[p] = NULL;
        g->wmap[p] = NULL;
        if (g->rambank_enabled) {
            g->rmap[p] = g->rambanks + (int64_t)g->rambank_selected * 8192 + (p - 0xA0) * 256;
            if (g->rambank_selected <= 3) g->wmap[p] = (uint8_t *)g->rmap[p];
        }
    }
}

static void update_maps(GB *g) {
    for (int p = 0; p < 256; p++) {
        g->rmap[p] = NULL;
        g->wmap[p] = NULL;
    }
    update_romx(g);
    for (int p = 0x80; p < 0xA0; p++) g->rmap[p] = g->wmap[p] = g->VRAM0 + (p - 0x80) * 256;
    update_rammaps(g);
    for (int p = 0xC0; p < 0xE0; p++) g->rmap[p] = g->wmap[p] = g->internal_ram0 + (p - 0xC0) * 256;
    for (int p = 0xE0; p < 0xFE; p++) g->rmap[p] = g->wmap[p] = g->internal_ram0 + (p - 0xE0) * 256;
}

/* ===================================================================== CPU (cpu.py + opcodes) */
/* CPU registers are held in a local struct for the duration of a cpu_tick batch so the compiler
 * can keep them in machine registers. The bus helpers below sync `cycles` back to the GB struct
 * before any slow path (IO ports catch up the LCD/timer/serial to g->cycles). Slow paths never
 * modify A..E, HL, SP, PC or cycles, so nothing has to be reloaded afterwards. */
typedef struct {
    int16_t A, F, B, C, D, E;
    int HL, SP, PC;
    int64_t cycles;
} CPURegs;

INLINE uint8_t RD(GB *g, CPURegs *r, uint16_t i) {
    if (i < 0x4000) {
        if (UNLIKELY(g->bootrom_enabled && i <= 0xFF)) return g->bootrom[i];
        return g->rom0[i];
    }
    if (i < 0x8000) return g->romx[i - 0x4000];
    const uint8_t *p = g->rmap[i >> 8];
    if (LIKELY(p != NULL)) return p[i & 0xFF];
    if (i >= 0xFF80 && i < 0xFFFF) return g->internal_ram1[i - 0xFF80];
    g->cycles = r->cycles;
    return mb_getitem_slow(g, i);
}

INLINE void WR(GB *g, CPURegs *r, uint16_t i, uint8_t value) {
    uint8_t *p = g->wmap[i >> 8];
    if (LIKELY(p != NULL)) {
        p[i & 0xFF] = value;
        return;
    }
    if (i >= 0xFF80 && i < 0xFFFF) {
        g->internal_ram1[i - 0xFF80] = value;
        return;
    }
    g->cycles = r->cycles;
    mb_setitem_slow(g, i, value);
}

INLINE uint8_t RD_IO(GB *g, CPURegs *r, uint16_t i) {
    if (LIKELY(i >= 0xFF80 && i < 0xFFFF)) return g->internal_ram1[i - 0xFF80];
    g->cycles = r->cycles;
    return mb_getitem_io_slow(g, i);
}

INLINE void WR_IO(GB *g, CPURegs *r, uint16_t i, uint8_t value) {
    if (LIKELY(i >= 0xFF80 && i < 0xFFFF)) {
        g->internal_ram1[i - 0xFF80] = value;
        return;
    }
    g->cycles = r->cycles;
    mb_setitem_io_slow(g, i, value);
}

#include "opcodes_gen.h"
#include "opcodes_threaded.h"

INLINE void handle_interrupt(GB *g, uint8_t flag, uint16_t addr) {
    g->interrupts_flag_register ^= flag;
    mb_setitem(g, (uint16_t)((g->SP - 1) & 0xFFFF), (uint8_t)(g->PC >> 8));
    mb_setitem(g, (uint16_t)((g->SP - 2) & 0xFFFF), (uint8_t)(g->PC & 0xFF));
    g->SP -= 2;
    g->SP &= 0xFFFF;
    g->PC = addr;
    g->cycles += 20;
    g->interrupt_master_enable = 0;
}

INLINE int check_interrupts(GB *g) {
    if (g->interrupt_queued) return 0;
    int raised_and_enabled = (g->interrupts_flag_register & 0x1F) & (g->interrupts_enabled_register & 0x1F);
    if (raised_and_enabled) {
        if (g->halted) {
            g->PC += 1;
            g->PC &= 0xFFFF;
        }
        if (g->interrupt_master_enable) {
            if (raised_and_enabled & INTR_VBLANK) handle_interrupt(g, INTR_VBLANK, 0x0040);
            else if (raised_and_enabled & INTR_LCDC) handle_interrupt(g, INTR_LCDC, 0x0048);
            else if (raised_and_enabled & INTR_TIMER) handle_interrupt(g, INTR_TIMER, 0x0050);
            else if (raised_and_enabled & INTR_SERIAL) handle_interrupt(g, INTR_SERIAL, 0x0058);
            else if (raised_and_enabled & INTR_HIGHTOLOW) handle_interrupt(g, INTR_HIGHTOLOW, 0x0060);
        }
        g->interrupt_queued = 1;
        return 1;
    } else {
        g->interrupt_queued = 0;
    }
    return 0;
}

/* Sets p to the 3 instruction bytes at PC (direct pointer into memory when that is equivalent to
 * three mb_getitem calls; otherwise copies them through the bus into buf). */
#define FETCH_PTR(g, r, p, buf)                                                       \
    do {                                                                              \
        int PC_ = (r)->PC;                                                            \
        if (!(g)->bootrom_enabled && PC_ + 2 < 0x4000) {                              \
            p = (g)->rom0 + PC_;                                                      \
        } else if (!(g)->bootrom_enabled && PC_ + 2 < 0x8000) {                       \
            p = (g)->romx + (PC_ - 0x4000);                                           \
        } else if (PC_ >= 0x8000 && (PC_ & 0xFF) <= 0xFD && (g)->rmap[PC_ >> 8]) {    \
            p = (g)->rmap[PC_ >> 8] + (PC_ & 0xFF);                                   \
        } else if (PC_ >= 0xFF80 && PC_ <= 0xFFFC) {                                  \
            p = (g)->internal_ram1 + (PC_ - 0xFF80);                                  \
        } else {                                                                      \
            buf[0] = RD(g, r, (uint16_t)PC_);                                         \
            buf[1] = RD(g, r, (uint16_t)((PC_ + 1) & 0xFFFF));                        \
            buf[2] = RD(g, r, (uint16_t)((PC_ + 2) & 0xFFFF));                        \
            p = buf;                                                                  \
        }                                                                             \
    } while (0)

INLINE void fetch_and_execute(GB *g, CPURegs *r) {
    const uint8_t *p;
    uint8_t buf[3];
    FETCH_PTR(g, r, p, buf);
#ifdef GB_PROFILE
    g->ioprof[p[0] == 0xCB ? 256 + p[1] : p[0]]++;
#endif
    dispatch_bytes(g, r, p);
}

#ifdef FB_THREADED
/* GCC cannot inline/clone a function whose static table stores label addresses. */
static __attribute__((noinline)) void cpu_tick(GB *g, int64_t cycles_target) {
#else
INLINE void cpu_tick(GB *g, int64_t cycles_target) {
#endif
    int64_t _cycles0 = g->cycles;
    int64_t _target = _cycles0 + cycles_target;
    if (check_interrupts(g)) g->halted = 0;
    if (g->halted && g->interrupt_queued) {
        g->halted = 0;
        g->PC += 1;
        g->PC &= 0xFFFF;
    } else if (g->halted) {
        PROF(2);
        g->cycles += cycles_target;
    }
    g->interrupt_queued = 0;
    g->bail = 0;
    if (g->cycles < _target) {
        CPURegs r;
        r.A = g->A; r.F = g->F; r.B = g->B; r.C = g->C; r.D = g->D; r.E = g->E;
        r.HL = g->HL; r.SP = g->SP; r.PC = g->PC; r.cycles = g->cycles;
#ifdef FB_THREADED
        {
            static const void *const T[256] = FB_DISPATCH_TABLE;
            const uint8_t *p;
            uint8_t buf[3];
#define FETCH() FETCH_PTR(g, &r, p, buf)
#define NEXT()                                         \
    do {                                               \
        if (g->bail || !(r.cycles < _target)) goto done; \
        FETCH();                                       \
        goto *T[p[0]];                                 \
    } while (0)
            FETCH();
            goto *T[p[0]];
            FB_HANDLERS
        done:;
#undef NEXT
#undef FETCH
        }
#else
        do {
            fetch_and_execute(g, &r);
            PROF(0);
        } while (!g->bail && r.cycles < _target);
#endif
        g->A = r.A; g->F = r.F; g->B = r.B; g->C = r.C; g->D = r.D; g->E = r.E;
        g->HL = r.HL; g->SP = r.SP; g->PC = r.PC; g->cycles = r.cycles;
    }
}

/* ===================================================================== Motherboard tick (mb.Motherboard.tick) */
static HOT int mb_tick(GB *g) {
    while (!g->frame_done) {
        int64_t mode0_cycles = MAX_CYCLES;
        /* min(timer, lcd, lcd_frame, sound(uint64), serial, mode0) with the Cython temp types */
        int64_t m = g->timer_cti;
        if (g->lcd_cti < m) m = g->lcd_cti;
        if (g->lcd_ctf < m) m = g->lcd_ctf;
        uint64_t u = (g->sound_cti < (uint64_t)m) ? g->sound_cti : (uint64_t)m;
        if ((uint64_t)g->serial_cti < u) u = (uint64_t)g->serial_cti;
        if ((uint64_t)mode0_cycles < u) u = (uint64_t)mode0_cycles;
        if (!(u > 4)) u = 4;
        int64_t cycles_target = (int64_t)u;
        if (g->breakpoint_singlestep) cycles_target = 4;
        PROF(1);
        cpu_tick(g, cycles_target);

        sound_tick(g, (uint64_t)g->cycles);
        if (serial_tick(g, (uint64_t)g->cycles)) g->interrupts_flag_register |= INTR_SERIAL;
        if (timer_tick(g, (uint64_t)g->cycles)) g->interrupts_flag_register |= INTR_TIMER;
        uint8_t li = lcd_tick(g, (uint64_t)g->cycles);
        if (li) g->interrupts_flag_register |= li;
        if (g->breakpoint_singlestep) break;
    }
    return g->breakpoint_singlestep;
}

/* ===================================================================== Breakpoints (mb.breakpoint_*) */
static uint8_t *bp_location(GB *g, int bank, int addr) {
    if (addr < 0x100 && bank == -1) return &g->bootrom[addr];
    if (addr < 0x4000) {
        if (g->external_rom_count < bank) return NULL;
        return &g->rombanks[(int64_t)bank * 0x4000 + addr];
    }
    if (addr < 0x8000) {
        if (g->external_rom_count < bank) return NULL;
        return &g->rombanks[(int64_t)bank * 0x4000 + addr - 0x4000];
    }
    if (addr < 0xA000) return bank == 0 ? &g->VRAM0[addr - 0x8000] : NULL;
    if (addr < 0xC000) {
        if (g->external_ram_count < bank) return NULL;
        return &g->rambanks[(int64_t)bank * 8192 + addr - 0xA000];
    }
    if (addr < 0xE000) return &g->internal_ram0[addr - 0xC000];
    return NULL;
}

int gb_breakpoint_find(GB *g, int bank, int addr) {
    for (int k = 0; k < g->n_breakpoints; k++)
        if (g->breakpoints[k].bank == bank && g->breakpoints[k].addr == addr) return k;
    return -1;
}

int gb_breakpoint_add(GB *g, int bank, int addr) {
    uint8_t *p = bp_location(g, bank, addr);
    if (!p) return -1;
    int opcode = *p;
    *p = OPCODE_BRK;
    int k = gb_breakpoint_find(g, bank, addr);
    if (k < 0) {
        if (g->n_breakpoints >= GB_MAX_BREAKPOINTS) return -2;
        k = g->n_breakpoints++;
    }
    g->breakpoints[k].bank = bank;
    g->breakpoints[k].addr = addr;
    g->breakpoints[k].opcode = opcode;
    return opcode;
}

int gb_breakpoint_remove(GB *g, int bank, int addr) {
    int k = gb_breakpoint_find(g, bank, addr);
    if (k < 0) return -1;
    uint8_t *p = bp_location(g, bank, addr);
    if (p) *p = (uint8_t)g->breakpoints[k].opcode;
    g->breakpoints[k] = g->breakpoints[g->n_breakpoints - 1];
    g->n_breakpoints--;
    return 0;
}

static void breakpoint_reinject(GB *g) {
    if (g->breakpoint_waiting < 0) return;
    int bank = (int)((g->breakpoint_waiting >> 24) & 0xFF);
    if (bank == 0xFF) bank = -1;
    int addr = (int)((g->breakpoint_waiting >> 8) & 0xFFFF);
    gb_breakpoint_add(g, bank, addr);
    g->breakpoint_waiting = -1;
}

/* returns 1 and fills bank/addr if a registered breakpoint is at PC */
static int breakpoint_reached(GB *g, int *bank_out, int *addr_out) {
    int pc = g->PC;
    int bank;
    if (pc < 0x100 && g->bootrom_enabled) bank = -1;
    else if (pc < 0x4000) bank = 0;
    else if (pc < 0x8000) bank = g->rombank_selected;
    else if (0xA000 <= pc && pc < 0xC000) bank = g->rambank_selected;
    else if (0xC000 <= pc) bank = 0;
    else return 0;
    int k = gb_breakpoint_find(g, bank, pc);
    if (k >= 0) {
        int opcode = g->breakpoints[k].opcode;
        g->breakpoint_waiting = (int64_t)((bank & 0xFF) << 24 | (pc & 0xFFFF) << 8 | (opcode & 0xFF));
        *bank_out = bank;
        *addr_out = pc;
        return 1;
    }
    return 0;
}

/* ===================================================================== Frame driver (pyboy.PyBoy.tick/_tick) */
int gb_send_input(GB *g, int event, int delay) {
    if (delay) {
        if (g->n_queued >= GB_MAX_EVENTS) return -1;
        int64_t frame = g->frame_count + delay;
        /* keep sorted by (frame, event): identical pop order to heapq on tuples */
        int k = g->n_queued;
        while (k > 0 && (g->queued[k - 1].frame > frame || (g->queued[k - 1].frame == frame && g->queued[k - 1].event > event))) {
            g->queued[k] = g->queued[k - 1];
            k--;
        }
        g->queued[k].frame = frame;
        g->queued[k].event = event;
        g->n_queued++;
    } else {
        if (g->n_events >= GB_MAX_EVENTS) return -1;
        g->events[g->n_events++] = event;
    }
    return 0;
}

static void frame_begin(GB *g, int render, int sound) {
    /* _handle_events */
    for (int k = 0; k < g->n_events; k++) {
        int ev = g->events[k];
        if (ev >= 1 && ev <= 16) {
            if (interaction_key_event(g, ev)) g->interrupts_flag_register |= INTR_HIGHTOLOW;
        }
        /* other events (PASS, etc.) are no-ops for the headless emulator */
    }
    g->n_events = 0; /* the list object is replaced in _post_handle_events */
    g->frame_done = 0;
    g->disable_renderer = !render;
    g->disable_sampling = !sound;
    g->audiobuffer_head = 0;
}

static void frame_end(GB *g) {
    g->frame_count += 1;
    /* _post_handle_events */
    g->n_events = 0;
    int k = 0;
    while (k < g->n_queued && g->queued[k].frame == g->frame_count) {
        g->events[g->n_events++] = g->queued[k].event;
        k++;
    }
    if (k) {
        memmove(g->queued, g->queued + k, (size_t)(g->n_queued - k) * sizeof(GBQueued));
        g->n_queued -= k;
    }
}

/* Runs the remainder of the current frame. Returns GB_TICK_HOOK when a hook must be dispatched
 * (breakpoint_waiting identifies it); call again to continue. */
static int frame_run(GB *g) {
    if (g->tick_resume_mode == 2) {
        /* resumed after a latch-triggered hook in the else-branch */
        g->breakpoint_singlestep = g->breakpoint_singlestep_latch;
    }
    g->tick_resume_mode = 0;
    while (mb_tick(g)) {
        breakpoint_reinject(g);
        int bank, addr;
        if (breakpoint_reached(g, &bank, &addr)) {
            gb_breakpoint_remove(g, bank, addr);
            g->breakpoint_singlestep_latch = 0;
            g->tick_resume_mode = 1;
            return GB_TICK_HOOK;
        } else {
            if (g->breakpoint_singlestep_latch) {
                g->tick_resume_mode = 2;
                return GB_TICK_HOOK;
            }
            g->breakpoint_singlestep = g->breakpoint_singlestep_latch;
        }
    }
    frame_end(g);
    return GB_TICK_DONE;
}

static int tick_loop(GB *g) {
    for (;;) {
        if (!g->tick_in_frame) {
            if (g->tick_remaining == 0) return GB_TICK_DONE;
            int last = g->tick_remaining == 1;
            frame_begin(g, g->tick_render && last, g->tick_sound && last);
            g->tick_in_frame = 1;
        }
        if (frame_run(g) == GB_TICK_HOOK) return GB_TICK_HOOK;
        g->tick_in_frame = 0;
        g->tick_remaining -= 1;
    }
}

int gb_tick(GB *g, int count, int render, int sound) {
    g->tick_remaining = count;
    g->tick_render = render;
    g->tick_sound = sound;
    g->tick_in_frame = 0;
    g->tick_resume_mode = 0;
    return tick_loop(g);
}

int gb_tick_resume(GB *g) { return tick_loop(g); }

uint8_t gb_getitem(GB *g, uint16_t i) { return mb_getitem(g, i); }
void gb_setitem(GB *g, uint16_t i, uint8_t v) { mb_setitem(g, i, v); }

/* ===================================================================== Init (constructors) */
static const uint8_t TONE_DEFAULT_LT = 64;

static void put64(uint8_t *p, uint64_t v) {
    for (int k = 0; k < 8; k++) p[k] = (uint8_t)(v >> (8 * k));
}

static void sound_channels_default(GB *g) {
    /* ToneChannel: 5 bytes, 16-bit period, 2 bytes, 6x 64-bit */
    uint8_t tone[57];
    memset(tone, 0, sizeof tone);
    put64(tone + 9, TONE_DEFAULT_LT); /* lengthtimer */
    put64(tone + 33, 4);              /* period */
    memcpy(g->ch_tone, tone, 57);
    memcpy(g->ch_sweep, tone, 57);
    memset(g->ch_sweep + 57, 0, 20);
    /* WaveChannel */
    memset(g->ch_wave, 0, sizeof g->ch_wave);
    memset(g->ch_wave, 0xFF, 16);
    put64(g->ch_wave + 23, 256); /* lengthtimer */
    put64(g->ch_wave + 39, 4);   /* period */
    /* NoiseChannel */
    memset(g->ch_noise, 0, sizeof g->ch_noise);
    put64(g->ch_noise + 9, 64);      /* lengthtimer */
    put64(g->ch_noise + 33, 8);      /* period */
    put64(g->ch_noise + 41, 1);      /* shiftregister */
    put64(g->ch_noise + 49, 0x4000); /* lfsrfeed */
}

int gb_init(GB *g, const uint8_t *rom, size_t rom_len, const uint8_t *bootrom, size_t bootrom_len) {
    memset(g, 0, sizeof *g);
    if (rom_len == 0 || rom_len % 0x4000 != 0) return -1;
    /* one bank of zero padding: PyBoy's unchecked memoryview indexing allows bank == count */
    g->rombanks = (uint8_t *)calloc(rom_len + 0x4000, 1);
    if (!g->rombanks) return -2;
    memcpy(g->rombanks, rom, rom_len);
    g->external_rom_count = (int)(rom_len / 0x4000);
    static const int ram_table[6] = {1, 1, 1, 4, 16, 8};
    uint8_t ram_code = rom[0x149];
    g->external_ram_count = ram_code < 6 ? ram_table[ram_code] : 1;
    g->carttype = rom[0x147];
    if (g->carttype < 0x0F || g->carttype > 0x13) return -3; /* MBC3 only */
    if (g->carttype == 0x0F || g->carttype == 0x10) return -4; /* RTC unsupported */
    if (rom[0x143] >> 7) return -5;                            /* CGB unsupported */
    if (bootrom_len != 0x100) return -6;
    memcpy(g->bootrom, bootrom, 0x100);

    g->rombank_selected = 1;
    g->bootrom_enabled = 1;
    g->breakpoint_waiting = -1;

    g->SB = 0xFF;
    g->serial_clock_target = MAX_CYCLES;
    g->directional = 0xF;
    g->standard = 0xF;

    lcdc_set(g, 0);
    g->stat_value = 0x80;
    g->stat_mode = 0;
    g->clock_target = FRAME_CYCLES;
    g->lcd_ctf = FRAME_CYCLES;
    g->next_stat_mode = 2;
    static const uint32_t rgb[4] = {0xFFFFFFFFu, 0xFF999999u, 0xFF555555u, 0xFF000000u};
    palette_init(&g->BGP, 0xFC, rgb);
    palette_init(&g->OBP0, 0xFF, rgb);
    palette_init(&g->OBP1, 0xFF, rgb);
    g->ly_window = 0;

    g->samples_per_frame = 48000 / 60;
    g->cycles_per_sample = (double)FRAME_CYCLES / (double)g->samples_per_frame;
    g->sound_cycles_target = (double)MAX_CYCLES;
    g->sound_cycles_target_512Hz = (double)MAX_CYCLES;
    g->sound_cti = MAX_CYCLES;
    sound_channels_default(g);
    update_maps(g);
    return 0;
}

void gb_free(GB *g) {
    free(g->rombanks);
    g->rombanks = NULL;
}

/* ===================================================================== Save / load state */
typedef struct {
    uint8_t *p;
    size_t n;
} W;
INLINE void w8(W *w, uint64_t v) {
    if (w->p) w->p[w->n] = (uint8_t)v;
    w->n++;
}
INLINE void w16(W *w, uint64_t v) {
    w8(w, v & 0xFF);
    w8(w, (v >> 8) & 0xFF);
}
INLINE void w32(W *w, uint64_t v) {
    for (int k = 0; k < 4; k++) w8(w, (v >> (8 * k)) & 0xFF);
}
INLINE void w64(W *w, uint64_t v) {
    for (int k = 0; k < 8; k++) w8(w, (v >> (8 * k)) & 0xFF);
}
static void wbytes(W *w, const uint8_t *b, size_t n) {
    if (w->p) memcpy(w->p + w->n, b, n);
    w->n += n;
}
static void wdouble(W *w, double d) {
    uint8_t b[8];
    memcpy(b, &d, 8);
    wbytes(w, b, 8);
}

static void save_all(GB *g, W *w) {
    w8(w, GB_STATE_VERSION);
    w8(w, g->bootrom_enabled);
    w8(w, g->key1);
    w8(w, g->double_speed);
    w8(w, g->cgb);
    /* cpu */
    w8(w, g->A & 0xFF); w8(w, g->F & 0xFF); w8(w, g->B & 0xFF);
    w8(w, g->C & 0xFF); w8(w, g->D & 0xFF); w8(w, g->E & 0xFF);
    w16(w, (uint16_t)g->HL); w16(w, (uint16_t)g->SP); w16(w, (uint16_t)g->PC);
    w8(w, g->interrupt_master_enable); w8(w, g->halted); w8(w, g->stopped);
    w8(w, g->interrupts_enabled_register); w8(w, g->interrupt_queued);
    w8(w, g->interrupts_flag_register);
    w64(w, (uint64_t)g->cycles);
    /* lcd */
    wbytes(w, g->VRAM0, sizeof g->VRAM0);
    wbytes(w, g->OAM, sizeof g->OAM);
    w8(w, g->lcdc_value); w8(w, g->BGP.value); w8(w, g->OBP0.value); w8(w, g->OBP1.value);
    w8(w, g->stat_value); w8(w, g->LY); w8(w, g->LYC);
    w8(w, g->SCY); w8(w, g->SCX); w8(w, g->WY); w8(w, g->WX);
    wbytes(w, &g->scanlineparameters[0][0], sizeof g->scanlineparameters);
    w8(w, g->cgb); w8(w, g->speed_shift); w8(w, g->frame_done); w8(w, g->first_frame); w8(w, g->reset);
    w64(w, g->lcd_last_cycles); w64(w, g->clock); w64(w, g->clock_target);
    w8(w, g->next_stat_mode);
    /* sound */
    w64(w, g->audiobuffer_head); w64(w, g->samples_per_frame); wdouble(w, g->cycles_per_sample);
    wbytes(w, g->audiobuffer, sizeof g->audiobuffer);
    w8(w, g->sound_speed_shift);
    wdouble(w, g->sound_cycles_target); wdouble(w, g->sound_cycles_target_512Hz);
    w64(w, g->sound_cti); w64(w, g->sound_cycles); w64(w, g->sound_last_cycles);
    w64(w, g->div_apu_counter); w64(w, g->div_apu);
    w8(w, g->poweron); w8(w, g->disable_sampling);
    wbytes(w, g->lr_flags, 8);
    w8(w, g->NR50);
    wbytes(w, g->ch_sweep, sizeof g->ch_sweep); wbytes(w, g->ch_tone, sizeof g->ch_tone);
    wbytes(w, g->ch_wave, sizeof g->ch_wave); wbytes(w, g->ch_noise, sizeof g->ch_noise);
    /* renderer */
    for (int k = 0; k < GB_ROWS * GB_COLS; k++) {
        w32(w, g->screenbuffer[k]);
        w8(w, g->screenbuffer_attributes[k]);
    }
    /* ram */
    wbytes(w, g->internal_ram0, sizeof g->internal_ram0);
    wbytes(w, g->non_io_internal_ram0, sizeof g->non_io_internal_ram0);
    wbytes(w, g->io_ports, sizeof g->io_ports);
    wbytes(w, g->internal_ram1, sizeof g->internal_ram1);
    wbytes(w, g->non_io_internal_ram1, sizeof g->non_io_internal_ram1);
    /* timer */
    w8(w, g->DIV); w8(w, g->TIMA); w16(w, g->DIV_counter); w16(w, g->TIMA_counter);
    w8(w, g->TMA); w8(w, g->TAC); w64(w, g->timer_last_cycles); w64(w, (uint64_t)g->timer_cti);
    /* cartridge */
    w8(w, g->rombank_selected); w8(w, g->rambank_selected); w8(w, g->rambank_enabled); w8(w, g->memorymodel);
    wbytes(w, g->rambanks, (size_t)g->external_ram_count * 8192);
    /* interaction */
    w8(w, g->directional); w8(w, g->standard);
    /* serial */
    w8(w, g->SB); w8(w, g->SC); w8(w, g->transfer_enabled); w8(w, g->internal_clock);
    w64(w, g->serial_last_cycles); w64(w, (uint64_t)g->serial_cti);
    w64(w, g->serial_clock); w64(w, g->serial_clock_target);
}

size_t gb_state_size(GB *g) {
    W w = {NULL, 0};
    save_all(g, &w);
    return w.n;
}

size_t gb_save_state(GB *g, uint8_t *out) {
    W w = {out, 0};
    save_all(g, &w);
    return w.n;
}

typedef struct {
    const uint8_t *p;
    size_t n, i;
    int err;
} R;
INLINE uint8_t r8(R *r) {
    if (r->i >= r->n) {
        r->err = 1;
        return 0;
    }
    return r->p[r->i++];
}
INLINE uint16_t r16(R *r) {
    uint16_t a = r8(r), b = r8(r);
    return (uint16_t)(a | (b << 8));
}
INLINE uint32_t r32(R *r) {
    uint32_t a = r8(r), b = r8(r), c = r8(r), d = r8(r);
    return a | (b << 8) | (c << 16) | (d << 24);
}
INLINE uint64_t r64(R *r) {
    uint64_t v = 0;
    for (int k = 0; k < 8; k++) v |= (uint64_t)r8(r) << (8 * k);
    return v;
}
static void rbytes(R *r, uint8_t *b, size_t n) {
    for (size_t k = 0; k < n; k++) b[k] = r8(r);
}
static double rdouble(R *r) {
    uint8_t b[8];
    rbytes(r, b, 8);
    double d;
    memcpy(&d, b, 8);
    return d;
}

int gb_load_state(GB *g, const uint8_t *in, size_t len) {
    R rr = {in, len, 0, 0}, *r = &rr;
    int v = r8(r);
    if (v >= 2) g->bootrom_enabled = r8(r);
    else g->bootrom_enabled = v;
    if (v > GB_STATE_VERSION) return -10;
    if (v >= 8) {
        g->key1 = r8(r);
        g->double_speed = r8(r);
        int _cgb = r8(r);
        if (_cgb) return -11;
    }
    /* cpu */
    g->A = r8(r); g->F = r8(r); g->B = r8(r); g->C = r8(r); g->D = r8(r); g->E = r8(r);
    g->HL = r16(r); g->SP = r16(r); g->PC = r16(r);
    g->interrupt_master_enable = r8(r); g->halted = r8(r); g->stopped = r8(r);
    if (v >= 5) g->interrupts_enabled_register = r8(r);
    if (v >= 8) {
        g->interrupt_queued = r8(r);
        g->interrupts_flag_register = r8(r);
    }
    if (v >= 12) g->cycles = (int64_t)r64(r);
    /* lcd */
    rbytes(r, g->VRAM0, sizeof g->VRAM0);
    rbytes(r, g->OAM, sizeof g->OAM);
    lcd_set_lcdc(g, r8(r));
    palette_set(&g->BGP, r8(r));
    palette_set(&g->OBP0, r8(r));
    palette_set(&g->OBP1, r8(r));
    if (v >= 5) {
        uint8_t sv = r8(r);
        g->stat_value = sv;
        g->stat_mode = sv & 3;
        g->LY = r8(r);
        g->LYC = r8(r);
    }
    g->SCY = r8(r); g->SCX = r8(r); g->WY = r8(r); g->WX = r8(r);
    if (v >= 11) {
        for (int y = 0; y < GB_ROWS; y++) {
            g->scanlineparameters[y][0] = r8(r);
            g->scanlineparameters[y][1] = r8(r);
            g->scanlineparameters[y][2] = r8(r);
            g->scanlineparameters[y][3] = r8(r);
            if (v > 3) g->scanlineparameters[y][4] = r8(r);
        }
    }
    if (v >= 8) {
        int _cgb = r8(r);
        if (_cgb) return -12;
        g->speed_shift = r8(r);
        if (v >= 13) {
            g->frame_done = r8(r);
            g->first_frame = r8(r);
            g->reset = r8(r);
        }
        if (v >= 12) g->lcd_last_cycles = r64(r);
        g->clock = r64(r);
        g->clock_target = r64(r);
        g->lcd_cti = (int64_t)((g->clock_target - g->clock) << g->speed_shift);
        g->lcd_ctf = (int64_t)(((uint64_t)FRAME_CYCLES - g->clock) << g->speed_shift);
        g->next_stat_mode = r8(r);
    }
    /* sound */
    if (v >= 8) {
        if (v == 13) {
            g->sound_last_cycles = r64(r);
            g->sound_cycles = r64(r);
            rbytes(r, g->ch_sweep, sizeof g->ch_sweep); rbytes(r, g->ch_tone, sizeof g->ch_tone);
            rbytes(r, g->ch_wave, sizeof g->ch_wave); rbytes(r, g->ch_noise, sizeof g->ch_noise);
        } else if (v >= 14) {
            g->audiobuffer_head = r64(r);
            uint64_t spf = r64(r);
            if (spf != g->samples_per_frame) return -13;
            g->cycles_per_sample = rdouble(r);
            rbytes(r, g->audiobuffer, sizeof g->audiobuffer);
            g->sound_speed_shift = r8(r);
            g->sound_cycles_target = rdouble(r);
            g->sound_cycles_target_512Hz = rdouble(r);
            g->sound_cti = r64(r);
            g->sound_cycles = r64(r);
            g->sound_last_cycles = r64(r);
            g->div_apu_counter = r64(r);
            g->div_apu = r64(r);
            g->poweron = r8(r);
            g->disable_sampling = r8(r);
            rbytes(r, g->lr_flags, 8);
            g->NR50 = r8(r);
            rbytes(r, g->ch_sweep, sizeof g->ch_sweep); rbytes(r, g->ch_tone, sizeof g->ch_tone);
            rbytes(r, g->ch_wave, sizeof g->ch_wave); rbytes(r, g->ch_noise, sizeof g->ch_noise);
        }
    }
    /* renderer */
    if (2 <= v && v < 11) {
        for (int y = 0; y < GB_ROWS; y++) {
            r8(r); r8(r); r8(r); r8(r);
            if (v > 3) r8(r);
        }
    }
    if (v >= 6) {
        for (int k = 0; k < GB_ROWS * GB_COLS; k++) {
            g->screenbuffer[k] = r32(r);
            if (v >= 10) g->screenbuffer_attributes[k] = r8(r);
        }
    }
    /* ram */
    rbytes(r, g->internal_ram0, sizeof g->internal_ram0);
    rbytes(r, g->non_io_internal_ram0, sizeof g->non_io_internal_ram0);
    rbytes(r, g->io_ports, sizeof g->io_ports);
    rbytes(r, g->internal_ram1, sizeof g->internal_ram1);
    rbytes(r, g->non_io_internal_ram1, sizeof g->non_io_internal_ram1);
    if (v < 5) g->interrupts_enabled_register = r8(r);
    if (v >= 5) {
        g->DIV = r8(r); g->TIMA = r8(r);
        g->DIV_counter = r16(r); g->TIMA_counter = r16(r);
        g->TMA = r8(r); g->TAC = r8(r);
        if (v >= 12) g->timer_last_cycles = r64(r);
        if (v >= 13) g->timer_cti = (int64_t)r64(r);
    }
    /* cartridge */
    g->rombank_selected = r8(r);
    g->rambank_selected = r8(r);
    g->rambank_enabled = r8(r);
    g->memorymodel = r8(r);
    rbytes(r, g->rambanks, (size_t)g->external_ram_count * 8192);
    /* interaction */
    if (v >= 7) {
        g->directional = r8(r);
        g->standard = r8(r);
    } else {
        g->directional = 0xF;
        g->standard = 0xF;
    }
    if (v >= 15) {
        g->SB = r8(r); g->SC = r8(r);
        g->transfer_enabled = r8(r); g->internal_clock = r8(r);
        g->serial_last_cycles = r64(r);
        g->serial_cti = (int64_t)r64(r);
        g->serial_clock = r64(r);
        g->serial_clock_target = r64(r);
    }
    update_maps(g);
    return r->err ? -1 : 0;
}
