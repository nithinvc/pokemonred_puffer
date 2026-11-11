"""The byte-snapshot flag/party classes and the fast screen packing must give exactly the values
the original ctypes / np.digitize code gave (no ROM needed)."""

from ctypes import c_uint8

import numpy as np
import pytest

from pokemonred_puffer.data.events import (
    EVENT_FLAGS_START,
    EVENTS,
    EventFlags,
    EventFlagsBits,
    events_mask,
)
from pokemonred_puffer.data.flags import Flags, FlagsBits
from pokemonred_puffer.data.missable_objects import MissableFlags, MissableFlagsBits
from pokemonred_puffer.data.party import PARTY_LENGTH_BYTES, PartyMons, PartyStruct, _PartyMonsUnion

SYMBOLS = {"wStatusFlags1": 0xD728, "wElite4Flags": 0xD734, "wPartyCount": 0xD163}
SYMBOLS["wPartyMons"] = 0xD16B


class FakeEmu:
    """Just enough of the emulator API for the snapshot classes: bytes memory + symbols."""

    def __init__(self, rng):
        self.memory = bytearray(rng.integers(0, 256, 0x10000, dtype=np.uint8).tobytes())

    def symbol_lookup(self, name):
        return 0, SYMBOLS[name]


def _ctypes_view(struct_cls, n, data):
    """What the original Union's `.b` field was: the bitfield struct over the same bytes."""
    return struct_cls.from_buffer_copy(bytes((c_uint8 * n)(*data)))


@pytest.mark.parametrize("seed", range(20))
def test_event_flags_match_ctypes(seed):
    emu = FakeEmu(np.random.default_rng(seed))
    ev = EventFlags(emu)
    ref = _ctypes_view(EventFlagsBits, 320, emu.memory[EVENT_FLAGS_START : EVENT_FLAGS_START + 320])
    names = [f[0] for f in EventFlagsBits._fields_]
    assert [ev.get_event(n) for n in names] == [getattr(ref, n) for n in names]
    assert list(ev.get_events(names)) == [getattr(ref, n) for n in names]
    assert ev.count_set(events_mask(EVENTS)) == sum(getattr(ref, n) for n in EVENTS)
    assert bytes(ev.asbytes) == bytes(emu.memory[EVENT_FLAGS_START : EVENT_FLAGS_START + 320])
    assert bytes(ev.b) == bytes(ref)


@pytest.mark.parametrize("seed", range(20))
def test_missable_flags_match_ctypes(seed):
    emu = FakeEmu(np.random.default_rng(100 + seed))
    mf = MissableFlags(emu)
    ref = _ctypes_view(MissableFlagsBits, 32, emu.memory[0xD5A6 : 0xD5A6 + 32])
    names = [f[0] for f in MissableFlagsBits._fields_]
    assert [mf.get_missable(n) for n in names] == [getattr(ref, n) for n in names]


@pytest.mark.parametrize("seed", range(20))
def test_flags_match_ctypes(seed):
    emu = FakeEmu(np.random.default_rng(200 + seed))
    fl = Flags(emu)
    start, end = SYMBOLS["wStatusFlags1"], SYMBOLS["wElite4Flags"] + 1
    ref = _ctypes_view(FlagsBits, 13, emu.memory[start:end])
    names = [f[0] for f in FlagsBits._fields_]
    assert [fl.get_bit(n) for n in names] == [bool(getattr(ref, n)) for n in names]


def test_set_methods_match_original_semantics():
    """set_* write the same RAM byte as before and update the snapshot like setattr did.

    The original writes `(mem & ~mask) | mask` with mask = value << bit, so setting False never
    clears the RAM bit; that quirk is preserved.
    """
    rng = np.random.default_rng(7)
    for cls, struct, n, setter, getter in [
        (EventFlags, EventFlagsBits, 320, "set_event", "get_event"),
        (MissableFlags, MissableFlagsBits, 32, "set_missable", "get_missable"),
        (Flags, FlagsBits, 13, "set_bit", "get_bit"),
    ]:
        names = [f[0] for f in struct._fields_]
        for _ in range(200):
            emu = FakeEmu(rng)
            obj = cls(emu)
            ref = struct.from_buffer_copy(bytes(obj.raw))
            name = names[int(rng.integers(0, len(names)))]
            value = bool(rng.integers(0, 2))
            before = bytes(emu.memory)
            getattr(obj, setter)(name, value)
            setattr(ref, name, int(value))
            assert bytes(obj.b) == bytes(ref), (cls.__name__, name, value)
            assert getattr(obj, getter)(name) == type(getattr(obj, getter)(name))(
                getattr(ref, name)
            )
            changed = [i for i in range(len(before)) if before[i] != emu.memory[i]]
            assert len(changed) <= 1


@pytest.mark.parametrize("seed", range(20))
def test_party_matches_ctypes(seed):
    emu = FakeEmu(np.random.default_rng(300 + seed))
    party = PartyMons(emu)
    start = SYMBOLS["wPartyMons"]
    ref = _PartyMonsUnion()
    ref.asbytes = (c_uint8 * PARTY_LENGTH_BYTES)(*emu.memory[start : start + PARTY_LENGTH_BYTES])
    assert party.party_size == emu.memory[SYMBOLS["wPartyCount"]]
    arr = party.np
    for i in range(6):
        for name, ctype in PartyStruct._fields_:
            want = getattr(ref.party[i], name)
            if hasattr(ctype, "_length_"):
                assert list(arr[name][i]) == list(want)
                assert list(getattr(party[i], name)) == list(want)
            else:
                assert int(arr[name][i]) == want, (i, name)
                assert getattr(party[i], name) == want


class _ScreenStub:
    """The attributes RedGymEnv.screen_obs and _screen_obs_reference read."""

    from pokemonred_puffer.environment import RedGymEnv

    screen_obs = RedGymEnv.screen_obs
    _screen_obs_reference = RedGymEnv._screen_obs_reference

    def __init__(self, rng, in_battle, near_edge, sweep=None):
        from pokemonred_puffer.global_map import GLOBAL_MAP_SHAPE

        self.reduce_res = True
        self.two_bit = True
        self.use_global_map = False
        palette = np.array([0, 85, 153, 255], dtype=np.uint8)
        px = palette[rng.integers(0, 4, (144, 160, 4))]
        if rng.random() < 0.5:  # any byte value, not only the 4 palette shades
            px = rng.integers(0, 256, (144, 160, 4), dtype=np.uint8)
        self.screen = type("S", (), {"ndarray": px})()
        self.explore_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.float32)
        hits = rng.integers(0, 2, GLOBAL_MAP_SHAPE).astype(bool) & (
            rng.random(GLOBAL_MAP_SHAPE) < 0.3
        )
        # forgetting factors leave arbitrary values in (0, 1]
        self.explore_map[hits] = np.where(
            rng.random(hits.sum()) < 0.5,
            rng.choice(np.array([0.15, 0.3, 0.5, 0.75, 1.0], dtype=np.float32), hits.sum()),
            rng.random(hits.sum()).astype(np.float32),
        )
        self._battle = in_battle
        self._coords = (
            (0, 0, 0) if near_edge else (int(rng.integers(5, 30)), int(rng.integers(5, 30)), 0)
        )
        if sweep is not None:  # the 10x10 window around the player holds every byte value 255 * v
            from pokemonred_puffer.global_map import local_to_global

            gr, gc = local_to_global(self._coords[1], self._coords[0], 0)
            k = (np.arange(100) + 100 * sweep) % 256
            self.explore_map[gr - 4 : gr + 6, gc - 4 : gc + 6] = (k / 255).reshape(10, 10)

    def read_m(self, addr):
        assert addr == 0xD057
        return self._battle

    def get_game_coords(self):
        return self._coords


@pytest.mark.parametrize("seed", range(33))
def test_screen_obs_fast_path_matches_reference(seed):
    rng = np.random.default_rng(seed)
    sweep = seed - 30 if seed >= 30 else None
    near_edge = sweep is None and seed % 7 == 0
    stub = _ScreenStub(rng, int(sweep is None and seed % 5 == 0), near_edge, sweep)
    fast, ref = stub.screen_obs(), stub._screen_obs_reference()
    assert fast.keys() == ref.keys()
    for k in ref:
        assert fast[k].dtype == ref[k].dtype and fast[k].shape == ref[k].shape, k
        assert np.array_equal(fast[k], ref[k]), k
