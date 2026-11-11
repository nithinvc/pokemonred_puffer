# Tests

The emulator and env speedups on this branch promise one thing: **the same config produces
bit-identical rollouts, only faster.** These tests check that promise at three levels.

```bash
uv pip install -e .                           # builds the FastBoy C extension
uv run --with pytest pytest                   # unit tests only (~2 s); ROM tests are skipped
POKEMON_RED_ROM=/path/to/red.gb uv run --with pytest pytest          # everything (~40 s)
POKEMON_RED_ROM=... PARITY_EMULATORS=fastboy,pyboy uv run --with pytest pytest   # + PyBoy (~2.5 min)
```

Run pytest from the repo root (the package is imported from there, as `train.py` does;
`pyproject.toml` sets `pythonpath = ["."]`). After editing `gbcore.c` or `_core.pyx`, rebuild
with `python setup.py build_ext --inplace` or the old extension keeps running.

## What each file covers

| File | Needs ROM | What it checks |
|---|---|---|
| `test_fastboy.py` | yes | FastBoy vs stock PyBoy 2.7 in lockstep from 12 save states (all five bundled state versions). After every env-shaped step: full save-state bytes, screen, and every hook callback (with LY and a RAM byte captured inside it). Plus memory-API parity (slices, banks, writes, symbols). `FASTBOY_LOCKSTEP_STEPS=1500` gives the 18k-step soak run. |
| `test_env_parity.py` | yes | Runs `parity_harness.py check` against `golden/env_parity.json.gz`, recorded from the original code (commit 24880ef). Two groups, `rollouts` and `unitfuzz`, per emulator. |
| `test_data_snapshots.py` | no | The byte-snapshot `EventFlags` / `MissableFlags` / `Flags` / `PartyMons` against the original ctypes structs on random RAM, including `set_*` semantics; the LUT screen packing against the original `np.digitize` path over every byte value. |
| `test_sum_tracked.py` | no | `SumTrackedArray` returns exactly `np.sum`'s float32 after random sequences of the writes the env and exploration wrappers make. |
| `test_environment.py` | no | Pre-existing unit test (PyBoy mocked). |

## The golden traces

`parity_harness.py` runs 60 scenarios with `configs/repro-gym2.yaml`:

- **one rollout per save state** (29): reset, then 1,000 random actions. Each step records a
  digest of every obs key, the reward as `float.hex`, terminated/truncated and an info digest.
  Every 100 steps it adds the full emulator save state and `agent_stats`.
- **`short_eps`**: `max_steps=150`, so episodes end and reset.
- **`swarm`**: loads another env's state halfway through via `update_state`.
- **`unitfuzz_<state>`** (29): 150 rounds of random but valid RAM pokes (bag, event flags,
  party, map, coordinates, battle flags), then the reward, observation, item, event and HM helpers
  are called directly and their results digested.

The committed golden stores one 8-byte digest per record (263 KB). It was recorded on macOS
arm64 and is byte-identical to a Linux x86_64 recording.

### When it fails

The compact golden tells you which scenario and record diverged first, not which field. For
field-level detail, record a full golden from the original code and check against it:

```bash
git worktree add ../baseline 24880ef
python tests/parity_harness.py record /tmp/gold_full.json.gz --repo ../baseline --full
python tests/parity_harness.py check  /tmp/gold_full.json.gz --group rollouts
```

### When behaviour changes on purpose

A change that is meant to alter rollouts (a reward fix, a new observation) fails this test by
design. Re-record the golden from the new code in the same commit, and say so in the
commit message:

```bash
POKEMON_RED_ROM=... python tests/parity_harness.py record tests/golden/env_parity.json.gz
```

### Things the golden depends on

- **Library versions.** The traces were recorded with numpy 1.23.3, PyBoy 2.7.0 and
  scikit-image 0.24.0 (stored in the golden's `_meta`). A numpy upgrade can change float
  summation and therefore rewards.
- **`PYTHONHASHSEED`.** `agent_stats` depends on set iteration order. The harness re-executes
  itself with `PYTHONHASHSEED=0`.
- **The env-id counter.** `RedGymEnv` numbers envs with a process-wide counter that
  `agent_stats` reports. The harness pins it per scenario, so groups can run on their own.
