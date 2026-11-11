"""FastBoy: drop-in replacement for the subset of the PyBoy 2.7 API used by RedGymEnv.

The emulator core (gbcore.c) is a C port of PyBoy's DMG core; emulation, screen output and
save-state bytes are identical to PyBoy 2.7.0 with sound_emulated=False. This module mirrors the
Python-level behavior PyBoy wraps around its core: the input event queue, hook dispatch, and the
PyBoyMemoryView slicing semantics (slices return lists).
"""

import os
import re

from ._core import Core

OPCODE_BRK = 0xDB
_BOOTROM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bootrom_dmg.bin")

# pyboy.utils.WindowEvent values
_PRESS = {"up": 1, "down": 2, "right": 3, "left": 4, "a": 5, "b": 6, "select": 7, "start": 8}
_RELEASE = {k: v + 8 for k, v in _PRESS.items()}


class _Screen:
    def __init__(self, core):
        self.ndarray = core.screen_ndarray


class _Memory:
    """Mirror of pyboy.PyBoyMemoryView for DMG."""

    def __init__(self, core):
        self._c = core

    @staticmethod
    def _fix_slice(addr):
        if addr.start is None:
            return (-1, 0, 0)
        if addr.stop is None:
            return (0, -1, 0)
        start, stop = addr.start, addr.stop
        if start > stop:
            return (-1, -1, 0)
        return start, stop, 1 if addr.step is None else addr.step

    def _range(self, addr):
        start, stop, step = self._fix_slice(addr)
        if not start >= 0:
            raise ValueError("Start address required")
        if not stop >= 0:
            raise ValueError("End address required")
        if not 0 <= start <= 0xFFFF:
            raise IndexError("Start address out of bounds")
        if not 0 <= stop <= 0x10000:
            raise IndexError("End address out of bounds")
        if not start < stop:
            raise ValueError("Start address has to come before end address")
        return start, stop, step

    def __getitem__(self, addr):
        c = self._c
        if isinstance(addr, tuple):
            bank, addr = addr
            if isinstance(addr, int):
                return self._bank_get(bank, addr, addr + 1, 1, True)
            start, stop, step = self._range(addr)
            return self._bank_get(bank, start, stop, step, False)
        if isinstance(addr, int):
            return c.getitem(addr)
        start, stop, step = self._range(addr)
        return c.getslice(start, stop, step)

    def _bank_get(self, bank, start, stop, step, single):
        c = self._c
        if start < 0x8000:
            if start >= 0x4000:
                start -= 0x4000
                stop -= 0x4000
            if not stop < 0x4000 and not single:
                raise IndexError("Out of bounds for reading ROM bank")
            if bank == -1:
                f = c.boot_byte
                vals = [f(x) for x in range(start, stop, step)]
            else:
                if not bank <= c.external_rom_count:
                    raise IndexError("ROM Bank out of range")
                vals = [c.rom_byte(bank, x) for x in range(start, stop, step)]
        elif start < 0xA000:
            if bank != 0:
                raise ValueError("Selecting bank of VRAM is only supported for CGB mode")
            vals = [c.vram_byte(x - 0x8000) for x in range(start, stop, step)]
        elif start < 0xC000:
            if not bank <= c.external_ram_count:
                raise IndexError("RAM Bank out of range")
            vals = [c.ram_byte(bank, x - 0xA000) for x in range(start, stop, step)]
        elif start < 0xE000:
            start -= 0xC000
            stop -= 0xC000
            if start >= 0x1000:
                start -= 0x1000
                stop -= 0x1000
            if bank != 0:
                raise ValueError("Selecting bank of WRAM is only supported for CGB mode")
            vals = [c.wram_byte(x + bank * 0x1000) for x in range(start, stop, step)]
        else:
            raise ValueError("Invalid memory address for bank")
        return vals[0] if single else vals

    def __setitem__(self, addr, v):
        c = self._c
        if isinstance(addr, tuple):
            bank, addr = addr
            if isinstance(addr, int):
                return self._bank_set(bank, addr, addr + 1, 1, [v])
            start, stop, step = self._range(addr)
            if hasattr(v, "__iter__"):
                v = list(v)
                if (stop - start) // step != len(v):
                    raise ValueError("slice does not match length of data")
            else:
                v = [v] * len(range(start, stop, step))
            return self._bank_set(bank, start, stop, step, v)
        if isinstance(addr, int):
            c.setitem(addr, v)
            return
        start, stop, step = self._range(addr)
        if hasattr(v, "__iter__"):
            if (stop - start) // step != len(v):
                raise ValueError("slice does not match length of data")
            it = iter(v)
            for x in range(start, stop, step):
                c.setitem(x, next(it))
        else:
            for x in range(start, stop, step):
                c.setitem(x, v)

    def _bank_set(self, bank, start, stop, step, vals):
        c = self._c
        xs = range(start, stop, step)
        if start < 0x8000:
            if start >= 0x4000:
                xs = range(start - 0x4000, stop - 0x4000, step)
            if bank == -1:
                raise NotImplementedError("bootrom writes are not supported")
            for x, val in zip(xs, vals):
                c.set_rom_byte(bank, x, val)
        elif start < 0xA000:
            if bank != 0:
                raise ValueError("Selecting bank of VRAM is only supported for CGB mode")
            for x, val in zip(xs, vals):
                c.set_vram_byte(x - 0x8000, val)
        elif start < 0xC000:
            for x, val in zip(xs, vals):
                c.set_ram_byte(bank, x - 0xA000, val)
        elif start < 0xE000:
            off = 0xC000 + (0x1000 if start - 0xC000 >= 0x1000 else 0)
            for x, val in zip(xs, vals):
                c.set_wram_byte(x - off + bank * 0x1000, val)
        else:
            raise ValueError("Invalid memory address for bank")


class FastBoy:
    def __init__(self, gamerom, window="null", symbols=None, sound_emulated=False, **kwargs):
        if sound_emulated:
            raise NotImplementedError("FastBoy does not emulate sound")
        if window not in ("null", "headless", "dummy"):
            raise NotImplementedError("FastBoy is headless only")
        if kwargs.get("cgb"):
            raise NotImplementedError("FastBoy is DMG only")
        self.gamerom = str(gamerom)
        with open(self.gamerom, "rb") as f:
            rom = f.read()
        with open(_BOOTROM, "rb") as f:
            boot = f.read()
        self._core = Core(rom, boot)
        ram_path = self.gamerom + ".ram"
        if os.path.isfile(ram_path):  # PyBoy loads a battery RAM file next to the ROM
            with open(ram_path, "rb") as f:
                self._core.load_cart_ram(f.read())
        self.memory = _Memory(self._core)
        # Fast accessors with memory-bus semantics (same values/side effects as `memory[...]`):
        # read_byte(addr) -> int, read_bytes(start, n) -> bytes
        self.read_byte = self._core.getitem
        self.read_bytes = self._core.read_bytes
        self.screen = _Screen(self._core)
        self.rom_symbols = {}
        self.rom_symbols_inverse = {}
        self.symbols_file = symbols
        self._load_symbols()
        self._hooks = {}
        self.stopped = False

    # ------------------------------------------------------------------ symbols
    def _load_symbols(self):
        gamerom_file_no_ext, rom_ext = os.path.splitext(self.gamerom)
        paths = [
            self.symbols_file,
            gamerom_file_no_ext + ".sym",
            gamerom_file_no_ext + rom_ext + ".sym",
        ]
        for sym_path in paths:
            if sym_path and os.path.isfile(sym_path):
                with open(sym_path) as f:
                    for _line in f.readlines():
                        line = _line.strip()
                        if line == "" or line.startswith(";") or line.startswith("["):
                            continue
                        try:
                            bank, addr, sym_label = re.split(":| ", line.strip())
                            bank = int(bank, 16)
                            addr = int(addr, 16)
                            self.rom_symbols.setdefault(bank, {}).setdefault(addr, []).append(
                                sym_label
                            )
                            self.rom_symbols_inverse[sym_label] = (bank, addr)
                        except ValueError:
                            pass

    def _lookup_symbol(self, symbol):
        bank_addr = self.rom_symbols_inverse.get(symbol)
        if bank_addr is None:
            raise ValueError("Symbol not found: %s" % symbol)
        return bank_addr

    def symbol_lookup(self, symbol):
        return self._lookup_symbol(symbol)

    # ------------------------------------------------------------------ emulation
    def tick(self, count=1, render=True, sound=True):
        if self.stopped:
            return False
        core = self._core
        r = core.tick(count, render, sound)
        while r:
            handler = self._hooks.get(core.breakpoint_waiting)
            if handler is not None:
                callback, context = handler
                callback(context)
            r = core.tick_resume()
        return True

    @property
    def frame_count(self):
        return self._core.frame_count

    def send_input(self, event, delay=0):
        if delay and not delay > 0:
            raise ValueError("Only positive integers allowed")
        self._core.send_input(int(event), int(delay))

    def button(self, input, delay=1):
        input = input.lower()
        if input not in _PRESS:
            raise ValueError("Unrecognized input")
        self.send_input(_PRESS[input])
        self.send_input(_RELEASE[input], delay)

    def button_press(self, input):
        self.send_input(_PRESS[input.lower()])

    def button_release(self, input):
        self.send_input(_RELEASE[input.lower()])

    def set_emulation_speed(self, target_speed):
        pass  # always unbounded; there is no window

    # ------------------------------------------------------------------ hooks
    def hook_register(self, bank, addr, callback, context):
        if bank is None and isinstance(addr, str):
            bank, addr = self._lookup_symbol(addr)
        opcode = self.memory[bank, addr]
        if opcode == OPCODE_BRK:
            raise ValueError("Hook already registered for this bank and address.")
        if self._core.breakpoint_add(bank, addr) < 0:
            raise IndexError("Unsupported breakpoint address")
        self._hooks[(bank & 0xFF) << 24 | (addr & 0xFFFF) << 8 | (opcode & 0xFF)] = (
            callback,
            context,
        )

    def hook_deregister(self, bank, addr):
        if bank is None and isinstance(addr, str):
            bank, addr = self._lookup_symbol(addr)
        k = self._core.breakpoint_find(bank, addr)
        if k < 0:
            raise ValueError("Breakpoint not found for bank and addr")
        # read the stored opcode before removal (memory currently holds BRK)
        self._core.breakpoint_remove(bank, addr)
        opcode = self.memory[bank, addr]
        self._hooks.pop((bank & 0xFF) << 24 | (addr & 0xFFFF) << 8 | (opcode & 0xFF))

    # ------------------------------------------------------------------ state
    def save_state(self, file_like_object):
        if isinstance(file_like_object, str):
            raise ValueError(
                "String not allowed. Did you specify a filepath instead of a file-like object?"
            )
        file_like_object.write(self._core.save_state())
        file_like_object.flush()

    def load_state(self, file_like_object):
        if isinstance(file_like_object, str):
            raise ValueError(
                "String not allowed. Did you specify a filepath instead of a file-like object?"
            )
        self._core.load_state(file_like_object.read())

    def stop(self, save=True):
        self.stopped = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()
