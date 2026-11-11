from ctypes import Structure, Union, c_uint16, c_uint8, sizeof

import numpy as np
from pyboy import PyBoy

from pokemonred_puffer.data.bitfields import read_ram, symbol_addr

from pokemonred_puffer.data.species import Species


class BoxStruct(Structure):
    _pack_ = 1
    _fields_ = [
        ("Species", c_uint8),
        ("HP", c_uint16),
        ("BoxLevel", c_uint8),
        ("Status", c_uint8),
        ("Type1", c_uint8),
        ("Type2", c_uint8),
        ("CatchRate", c_uint8),
        ("Moves", 4 * c_uint8),
        ("OTID", c_uint16),
        ("Exp", 3 * c_uint8),
        ("HPExp", c_uint16),
        ("AttackExp", c_uint16),
        ("DefenseExp", c_uint16),
        ("SpeedExp", c_uint16),
        ("SpecialExp", c_uint16),
        ("DVs", 2 * c_uint8),
        ("PP", 4 * c_uint8),
    ]


class PartyStruct(Structure):
    _pack_ = 1
    _fields_ = BoxStruct._fields_ + [
        ("Level", c_uint8),
        ("MaxHP", c_uint16),
        ("Attack", c_uint16),
        ("Defense", c_uint16),
        ("Speed", c_uint16),
        ("Special", c_uint16),
    ]


PARTY_LENGTH_BYTES = 6 * sizeof(PartyStruct)


class _PartyMonsUnion(Union):
    _fields_ = [("party", 6 * PartyStruct), ("asbytes", c_uint8 * PARTY_LENGTH_BYTES)]


def _numpy_dtype(struct_cls) -> np.dtype:
    names, formats, offsets = [], [], []
    for name, ctype in struct_cls._fields_:
        field = getattr(struct_cls, name)
        if hasattr(ctype, "_length_"):
            fmt = (np.dtype("u1"), (ctype._length_,))
        else:
            fmt = np.dtype({1: "u1", 2: "<u2"}[sizeof(ctype)])
        names.append(name)
        formats.append(fmt)
        offsets.append(field.offset)
    return np.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": sizeof(struct_cls)}
    )


# Same layout as PartyStruct (packed, native little-endian uint16 fields).
PARTY_DTYPE = _numpy_dtype(PartyStruct)


class PartyMons:
    """Snapshot of the 6 party structs (same API and values as the former ctypes Union).

    `np` is a structured numpy view of the same bytes; the ctypes view is built lazily.
    """

    __slots__ = ("party_size", "raw", "_union")

    def __init__(self, emu: PyBoy):
        wPartyMons = symbol_addr(emu, "wPartyMons")
        wPartyCount = symbol_addr(emu, "wPartyCount")
        self.party_size = emu.memory[wPartyCount]
        self.raw = read_ram(emu, wPartyMons, PARTY_LENGTH_BYTES)
        self._union = None

    @property
    def np(self) -> np.ndarray:
        return np.frombuffer(self.raw, dtype=PARTY_DTYPE)

    def _ctypes(self):
        if self._union is None:
            self._union = _PartyMonsUnion.from_buffer_copy(self.raw)
        return self._union

    @property
    def party(self):
        return self._ctypes().party

    @property
    def asbytes(self):
        return self._ctypes().asbytes

    def __getitem__(self, idx):
        return self.party[idx]

    def __repr__(self):
        return str([Species(x.Species).name for x in self.party[: self.party_size]])
