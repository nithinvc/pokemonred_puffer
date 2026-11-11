/*
 * gbcore: a single-translation-unit C port of PyBoy v2.7.0's DMG core.
 *
 * Semantics (including integer types, scheduling and quirks) mirror PyBoy's Cython build so that
 * emulation, rendering and save-state bytes are bit-identical. Only what Pokemon Red needs is
 * supported: DMG mode, MBC3 (no RTC), sound registers not emulated (sound_emulated=False).
 *
 * Derived from PyBoy (https://github.com/Baekalfen/PyBoy), LGPL-3.0.
 */
#ifndef GBCORE_H
#define GBCORE_H

#include <stdint.h>
#include <stddef.h>

#define GB_ROWS 144
#define GB_COLS 160
#define GB_FRAME_CYCLES 70224
#define GB_MAX_CYCLES (1LL << 31)
#define GB_STATE_VERSION 15
#define GB_MAX_BREAKPOINTS 128
#define GB_MAX_EVENTS 256

typedef struct {
    uint8_t value;
    uint32_t lookup[4];
    uint32_t palette_mem_rgb[4];
} GBPalette;

typedef struct {
    int bank, addr, opcode;
} GBBreakpoint;

typedef struct {
    int64_t frame;
    int event;
} GBQueued;

typedef struct GB {
    /* ---------------- CPU ---------------- */
    int16_t A, F, B, C, D, E;
    int HL, SP, PC;
    int interrupt_master_enable, interrupt_queued, halted, stopped, bail;
    uint8_t interrupts_flag_register, interrupts_enabled_register;
    int64_t cycles;

    /* ---------------- Motherboard ---------------- */
    int bootrom_enabled;
    uint8_t key1;
    int double_speed;
    int cgb;
    int breakpoint_singlestep, breakpoint_singlestep_latch;
    int64_t breakpoint_waiting;
    GBBreakpoint breakpoints[GB_MAX_BREAKPOINTS];
    int n_breakpoints;
    uint8_t bootrom[0x100];
    uint16_t serialbuffer_count;

    /* ---------------- Cartridge (MBC3) ---------------- */
    uint8_t *rombanks; /* external_rom_count * 0x4000 */
    int external_rom_count, external_ram_count;
    uint8_t rambanks[16 * 8192];
    int rambank_enabled;
    uint16_t rambank_selected, rombank_selected, rombank_selected_low;
    uint8_t memorymodel, carttype;

    /* ---------------- RAM ---------------- */
    uint8_t internal_ram0[8192];
    uint8_t non_io_internal_ram0[0x60];
    uint8_t io_ports[0x4C];
    uint8_t internal_ram1[0x7F];
    uint8_t non_io_internal_ram1[0x34];

    /* ---------------- Timer ---------------- */
    uint64_t DIV, TIMA, TMA, TAC, DIV_counter, TIMA_counter;
    int64_t timer_cti;
    uint64_t timer_last_cycles;

    /* ---------------- Serial ---------------- */
    uint64_t SB, SC;
    int64_t serial_cti;
    uint64_t serial_last_cycles, serial_clock, serial_clock_target;
    int transfer_enabled, internal_clock;

    /* ---------------- Interaction ---------------- */
    uint8_t directional, standard;

    /* ---------------- LCD ---------------- */
    int disable_renderer;
    uint8_t VRAM0[8192];
    uint8_t OAM[0xA0];
    uint8_t SCY, SCX, WY, WX, next_stat_mode;
    int frame_done, first_frame, reset;
    uint8_t LY, LYC;
    uint64_t clock, clock_target;
    /* LCDC register: the "bint" fields hold the raw masked value, as in the Cython build */
    uint8_t lcdc_value;
    int lcd_enable, windowmap_select, window_enable, tiledata_select, backgroundmap_select,
        sprite_height, sprite_enable, background_enable;
    uint16_t backgroundmap_offset, windowmap_offset;
    uint8_t stat_value, stat_mode;
    GBPalette BGP, OBP0, OBP1;
    uint8_t scanlineparameters[GB_ROWS][5];
    uint64_t lcd_last_cycles;
    int64_t lcd_cti, lcd_ctf;
    uint8_t speed_shift;

    /* ---------------- Renderer ---------------- */
    uint32_t screenbuffer[GB_ROWS * GB_COLS];
    uint8_t screenbuffer_attributes[GB_ROWS * GB_COLS];
    int ly_window;
    int sprites_to_render[10];

    /* ---------------- Sound (registers not emulated; state kept for save-state parity) ---------------- */
    uint64_t audiobuffer_head, samples_per_frame;
    double cycles_per_sample;
    uint8_t audiobuffer[1602];
    uint8_t sound_speed_shift;
    double sound_cycles_target, sound_cycles_target_512Hz;
    uint64_t sound_cti, sound_cycles, sound_last_cycles, div_apu_counter, div_apu;
    uint8_t poweron;
    int disable_sampling;
    uint8_t lr_flags[8];
    uint8_t NR50;
    uint8_t ch_sweep[77], ch_tone[57], ch_wave[63], ch_noise[65];

    /* ---------------- Fast-path page tables (derived state, rebuilt on bank switches) ---------------- */
    const uint8_t *rmap[256]; /* plain-memory reads; NULL -> slow path */
    uint8_t *wmap[256];       /* plain-memory writes without side effects; NULL -> slow path */
    const uint8_t *rom0, *romx; /* current ROM bank bases */
    uint64_t prof[16];
    uint64_t ioprof[512];

    /* ---------------- Frame driver (PyBoy._tick / tick) ---------------- */
    int64_t frame_count;
    int events[GB_MAX_EVENTS];
    int n_events;
    GBQueued queued[GB_MAX_EVENTS];
    int n_queued;
    /* resumable tick state */
    int tick_remaining, tick_render, tick_sound, tick_in_frame, tick_resume_mode;
} GB;

/* return codes of gb_tick / gb_tick_resume */
#define GB_TICK_DONE 0
#define GB_TICK_HOOK 1

int gb_init(GB *g, const uint8_t *rom, size_t rom_len, const uint8_t *bootrom, size_t bootrom_len);
void gb_free(GB *g);
int gb_tick(GB *g, int count, int render, int sound);
int gb_tick_resume(GB *g);

uint8_t gb_getitem(GB *g, uint16_t i);
void gb_setitem(GB *g, uint16_t i, uint8_t v);

int gb_breakpoint_add(GB *g, int bank, int addr);
int gb_breakpoint_remove(GB *g, int bank, int addr);
int gb_breakpoint_find(GB *g, int bank, int addr);

int gb_send_input(GB *g, int event, int delay);

size_t gb_state_size(GB *g);
size_t gb_save_state(GB *g, uint8_t *out);
int gb_load_state(GB *g, const uint8_t *in, size_t len);

#endif
