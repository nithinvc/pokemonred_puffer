"""Helpers for fast, exact snapshots of RAM regions that are described by ctypes bitfield structs.

The flag classes (EventFlags, MissableFlags, Flags, PartyMons) used to copy RAM into new ctypes
unions every step and read fields with getattr. They now keep the raw bytes and read fields with
precomputed bit positions. The positions are derived from the ctypes layout itself, so values are
identical to what the ctypes getattr returned.
"""

from ctypes import sizeof


def read_ram(emu, start: int, n: int) -> bytes:
    """`bytes(emu.memory[start:start + n])`, using the emulator's fast path when it has one."""
    read_bytes = getattr(emu, "read_bytes", None)
    if read_bytes is not None:
        return read_bytes(start, n)
    return bytes(emu.memory[start : start + n])


def field_layout(struct_cls) -> dict[str, tuple[int, int]]:
    """Map field name -> (bit shift, mask) within the struct's bytes read as a little-endian int."""
    layout = {}
    for field in struct_cls._fields_:
        name, ctype = field[0], field[1]
        width = field[2] if len(field) > 2 else 8 * sizeof(ctype)
        probe = struct_cls()
        setattr(probe, name, (1 << width) - 1)
        value = int.from_bytes(bytes(probe), "little")
        shift = (value & -value).bit_length() - 1
        assert value == ((1 << width) - 1) << shift, name
        layout[name] = (shift, (1 << width) - 1)
    return layout


def bit_positions(struct_cls) -> dict[str, tuple[int, int]]:
    """Map field name -> (byte index, bit index) for structs whose fields are all 1 bit wide."""
    out = {}
    for name, (shift, mask) in field_layout(struct_cls).items():
        assert mask == 1, f"{name} is not a 1-bit field"
        out[name] = (shift >> 3, shift & 7)
    return out


# id(emu) -> (emu, {name: addr}). Holding the emulator reference keeps its id from being reused.
_symbol_cache: dict[int, tuple[object, dict[str, int]]] = {}


def symbol_addr(emu, name: str) -> int:
    """Memoized `emu.symbol_lookup(name)[1]` (the symbol table never changes)."""
    entry = _symbol_cache.get(id(emu))
    if entry is None or entry[0] is not emu:
        entry = _symbol_cache[id(emu)] = (emu, {})
    table = entry[1]
    addr = table.get(name)
    if addr is None:
        addr = table[name] = emu.symbol_lookup(name)[1]
    return addr
