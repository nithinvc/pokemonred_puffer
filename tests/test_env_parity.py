"""Env rollouts must reproduce the golden recorded from the original code (see parity_harness.py).

Requires the Pokemon Red ROM: set POKEMON_RED_ROM=/path/to/red.gb.
Runs on FastBoy by default (~25 s). Set PARITY_EMULATORS=fastboy,pyboy to also check the new
Python env code on stock PyBoy (~100 s more).
"""

import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "golden", "env_parity.json.gz")
EMULATORS = [e for e in os.environ.get("PARITY_EMULATORS", "fastboy").split(",") if e]

pytestmark = pytest.mark.skipif(
    not os.environ.get("POKEMON_RED_ROM"), reason="set POKEMON_RED_ROM to run env parity tests"
)


@pytest.mark.parametrize("group", ["rollouts", "unitfuzz"])
@pytest.mark.parametrize("emulator", EMULATORS)
def test_env_parity(emulator, group):
    cmd = [sys.executable, os.path.join(HERE, "parity_harness.py"), "check", GOLDEN]
    cmd += ["--emulator", emulator, "--group", group]
    env = dict(os.environ, PYTHONHASHSEED="0")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    # the env prints "coord out of bounds!" warnings to stdout; keep only the harness verdict
    report = [ln for ln in proc.stdout.splitlines() if not ln.startswith("coord out of bounds")]
    assert proc.returncode == 0, "\n".join(report) + "\n" + proc.stderr[-2000:]
    assert "PARITY OK" in proc.stdout
