"""FastBoy must be bit-identical to PyBoy 2.7 (sound_emulated=False).

Runs both emulators in lockstep from save states with env-shaped inputs and compares, after every
step, the full save-state bytes (CPU, LCD, RAM, timers, cartridge, ...), the screen and the
sequence of hook callbacks (with LY and a RAM byte captured inside each callback).

Requires the Pokemon Red ROM: set POKEMON_RED_ROM=/path/to/red.gb.
FASTBOY_LOCKSTEP_STEPS (default 150) sets steps per state; 1500 gives the 18k-step soak run.
"""

import io
import os
import random

import numpy as np
import pytest

ROM = os.environ.get("POKEMON_RED_ROM")
STEPS = int(os.environ.get("FASTBOY_LOCKSTEP_STEPS", "150"))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYM = os.path.join(REPO, "pokemonred_puffer", "pokered.sym")
# Covers all five PyBoy save-state versions shipped in pyboy_states/ (7, 9, 10, 11, 12).
STATES = [
    "Bulbasaur",
    "cut",
    "mtmoon",
    "victory_road",
    "seafoam",
    "surf",
    "pokeflute",
    "rocktunnel",
    "game_corner",
    "init",
    "has_pokedex",
    "beat_lance_1",
]
# The hooks the env registers, plus a few extra labels.
HOOKS = [
    "DisplayStartMenu",
    "RedisplayStartMenu",
    "StartMenu_Item",
    "StartMenu_Pokemon",
    "StartMenu_Pokemon.choseStats",
    "StartMenu_Item.choseItem",
    "DisplayTextID.spriteHandling",
    "CheckForHiddenObject.foundMatchingObject",
    "HandleBlackOut",
    "SetLastBlackoutMap.done",
    "AnimateHealingMachine",
    "CheckWarpsNoCollisionLoop",
    "ItemUseBall.loop",
]
PRESS = [2, 4, 3, 1, 5, 6, 8]  # WindowEvent.PRESS_* for down, left, right, up, a, b, start

pytestmark = pytest.mark.skipif(not ROM, reason="set POKEMON_RED_ROM to run emulator parity tests")

# (section, size) of a PyBoy v15 save state for DMG / MBC3 with 4 RAM banks; only used to say
# where the first differing byte is.
_STATE_LAYOUT = [
    ("motherboard", 5),
    ("cpu", 26),
    ("lcd", 8192 + 160 + 11 + 720 + 5 + 24 + 1),
    ("sound", 8 * 14 + 1602 + 1 + 1 + 1 + 1 + 77 + 57 + 63 + 65),
    ("renderer", 144 * 160 * 5),
    ("ram", 8192 + 0x60 + 0x4C + 0x7F + 0x34),
    ("timer", 24),
    ("cartridge", 4 + 4 * 8192),
    ("interaction", 2),
    ("serial", 36),
]


def _where(off):
    o = 0
    for name, n in _STATE_LAYOUT:
        if off < o + n:
            return f"{name}+{off - o}"
        o += n
    return f"end+{off - o}"


def _make(cls, state, log):
    from pyboy import PyBoy

    if cls is PyBoy:
        pb = cls(ROM, window="null", log_level="CRITICAL", symbols=SYM, sound_emulated=False)
    else:
        pb = cls(ROM, window="null", symbols=SYM, sound_emulated=False)
    for h in HOOKS:
        pb.hook_register(
            None, h, lambda ctx, pb=pb: log.append((ctx, pb.memory[0xFF44], pb.memory[0xD35E])), h
        )
    bank, addr = pb.symbol_lookup("IsSpriteOrSignInFrontOfPlayer.retry")
    pb.hook_register(bank, addr - 1, lambda ctx: log.append(ctx), "sign")
    with open(os.path.join(REPO, "pyboy_states", f"{state}.state"), "rb") as f:
        pb.load_state(f)
    return pb


def _state(pb):
    f = io.BytesIO()
    pb.save_state(f)
    return f.getvalue()


def _state_diff(sa, sb):
    if len(sa) != len(sb):
        return f"length {len(sa)} vs {len(sb)}"
    diffs = [k for k in range(len(sa)) if sa[k] != sb[k]]
    return f"{len(diffs)} bytes differ, first at {[_where(k) for k in diffs[:6]]}"


@pytest.mark.parametrize("state", STATES)
def test_lockstep(state):
    from pyboy import PyBoy

    from pokemonred_puffer.fastboy import FastBoy

    la, lb = [], []
    a, b = _make(PyBoy, state, la), _make(FastBoy, state, lb)
    sa, sb = _state(a), _state(b)
    assert sa == sb, f"state differs right after load: {_state_diff(sa, sb)}"
    rng = random.Random(0)
    for i in range(STEPS):
        act, mode = rng.randrange(7), rng.random()
        for pb in (a, b):
            if mode < 0.1:  # dialogue-skip shape
                pb.button("a", 8)
                pb.tick(24, False)
            else:  # env action shape: press, release after 8 frames, render the last frame
                pb.send_input(PRESS[act])
                pb.send_input(PRESS[act] + 8, delay=8)
                pb.tick(23, render=False)
                pb.tick(1, render=True)
        assert la == lb, f"hook calls differ at step {i}: {la[-3:]} vs {lb[-3:]}"
        assert np.array_equal(a.screen.ndarray, b.screen.ndarray), f"screen differs at step {i}"
        sa, sb = _state(a), _state(b)
        assert sa == sb, f"save state differs at step {i}: {_state_diff(sa, sb)}"


@pytest.mark.parametrize("state", ["Bulbasaur", "surf"])
def test_memory_api(state):
    from pyboy import PyBoy

    from pokemonred_puffer.fastboy import FastBoy

    a, b = _make(PyBoy, state, []), _make(FastBoy, state, [])
    for pb in (a, b):
        pb.tick(60, render=True)
    assert a.memory[0xC000:0xC100] == b.memory[0xC000:0xC100]
    assert a.memory[0xFF80:0xFFFF] == b.memory[0xFF80:0xFFFF]
    assert a.memory[0xC000:0xC010:2] == b.memory[0xC000:0xC010:2]
    assert a.memory[0, 0x100:0x150] == b.memory[0, 0x100:0x150]  # ROM bank 0
    assert a.memory[5, 0x4000:0x4040] == b.memory[5, 0x4000:0x4040]  # switchable ROM bank
    assert a.memory[0xD35E] == b.memory[0xD35E] == b.read_byte(0xD35E)
    assert b.read_bytes(0xD000, 64) == bytes(a.memory[0xD000:0xD040])
    for name in ["wPartyCount", "wCurMap", "wBagItems"]:
        assert a.symbol_lookup(name) == b.symbol_lookup(name)
    # writes go through the same bus semantics
    for pb in (a, b):
        pb.memory[0xC100] = 0x42
        pb.memory[0xC101:0xC104] = [1, 2, 3]
        assert pb.memory[0xC100:0xC104] == [0x42, 1, 2, 3]
        pb.tick(1, render=False)
    assert a.memory[0xC100:0xC104] == b.memory[0xC100:0xC104]
    assert _state(a) == _state(b)
