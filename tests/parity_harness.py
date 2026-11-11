"""Bit-exact env rollout parity: record golden traces from one checkout, check another.

The golden in tests/golden/ was recorded from the original code (commit 24880ef: PyBoy 2.7.0 and
the original Python env). Any change that keeps "same config -> same rollouts" must reproduce it.

Usage (needs the Pokemon Red ROM):
  POKEMON_RED_ROM=red.gb python tests/parity_harness.py check  tests/golden/env_parity.json.gz
  POKEMON_RED_ROM=red.gb python tests/parity_harness.py record OUT.json.gz [--repo CHECKOUT] [--full]

Options:
  --repo PATH        run against another checkout (e.g. a `git worktree` of an older commit)
  --emulator NAME    set env.emulator (fastboy | pyboy); default: the config's value
  --group NAME       rollouts | unitfuzz | all (default all)
  --steps N          env steps per rollout scenario (default 1000; must match the golden)
  --full             record per-step records instead of digests (large, but `check` can then
                     say which obs key / field differs)

Scenarios (configs/repro-gym2.yaml, ObjectRewardRequiredEventsMapIdsFieldMoves reward,
OnResetExplorationWrapper):
  <state>            one per pyboy_states/*.state: reset, then N random actions. Per step: digest
                     of every obs key, reward (float.hex), terminated/truncated, info digest.
                     Every 100 steps: digest of the full emulator save state and of agent_stats.
  short_eps          max_steps=150, so episodes end and reset (exploration-wrapper reset path).
  swarm              mtmoon, with the final emulator state of `cut` loaded halfway via update_state.
  unitfuzz_<state>   150 rounds of random but valid RAM pokes (bag, event flags, party, map,
                     coordinates, battle flags), then the env's reward / obs / item / event / HM
                     helpers are called directly and their results digested.

agent_stats depends on set iteration order, so the harness re-executes itself with
PYTHONHASHSEED=0 when needed. It also reports the env's process-wide env_id counter, which the
harness pins per scenario, so any group can run on its own.
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import random
import sys
import time

THIS_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_STEPS = 1000
UNIT_FUZZ_ROUNDS = 150


def h(b: bytes) -> str:
    return hashlib.blake2b(b, digest_size=8).hexdigest()


def canon(x, out):
    """Deterministic serialization of nested info structures."""
    import numpy as np

    if isinstance(x, dict):
        out.append(b"{")
        for k, v in x.items():
            canon(k, out)
            canon(v, out)
        out.append(b"}")
    elif isinstance(x, (list, tuple)):
        out.append(b"[" if isinstance(x, list) else b"(")
        for v in x:
            canon(v, out)
        out.append(b"]")
    elif isinstance(x, np.ndarray):
        out.append(f"nd{x.dtype}{x.shape}".encode())
        out.append(np.ascontiguousarray(x).tobytes())
    elif isinstance(x, (bytes, bytearray)):
        out.append(b"B" + h(bytes(x)).encode())
    elif isinstance(x, float):
        out.append(b"f" + x.hex().encode())
    elif isinstance(x, np.floating):
        out.append(b"F" + float(x).hex().encode())
    else:
        out.append(repr(x).encode())
    return out


def canon_digest(x) -> str:
    return h(b"".join(canon(x, [])))


def obs_digest(obs):
    import numpy as np

    return {
        k: h(np.ascontiguousarray(v).tobytes() + str(v.dtype).encode() + str(v.shape).encode())
        for k, v in obs.items()
    }


def record_digest(rec) -> str:
    """Compact form of one golden record (what the committed golden stores)."""
    return h(json.dumps(rec, sort_keys=True).encode())


class Runner:
    def __init__(self, repo: str, rom: str, emulator: str | None):
        self.repo, self.rom, self.emulator = repo, rom, emulator
        sys.path.insert(0, repo)
        os.chdir(repo)  # the env resolves pokered.sym, pyboy_states and configs relative to cwd
        self.states = sorted(p[:-6] for p in os.listdir("pyboy_states") if p.endswith(".state"))

    @staticmethod
    def set_env_counter(n):
        """RedGymEnv numbers envs with a process-wide counter (shared memory) and agent_stats
        reports the number, so each scenario pins it to its index in the canonical order. That
        makes a scenario's trace independent of which scenarios ran before it in the process."""
        from pokemonred_puffer.environment import RedGymEnv

        RedGymEnv.env_id.buf[:4] = n.to_bytes(4, "big")

    def make_env(self, init_state, env_index, **over):
        self.set_env_counter(env_index)
        from omegaconf import OmegaConf

        from pokemonred_puffer.rewards.baseline import ObjectRewardRequiredEventsMapIdsFieldMoves
        from pokemonred_puffer.wrappers.exploration import OnResetExplorationWrapper

        cfg = OmegaConf.load(os.path.join(self.repo, "configs", "repro-gym2.yaml"))
        cfg.env.gb_path = self.rom
        cfg.env.init_state = init_state
        if self.emulator is not None:
            cfg.env.emulator = self.emulator
        for k, v in over.items():
            cfg.env[k] = v
        reward_cfg = cfg.rewards["baseline.ObjectRewardRequiredEventsMapIdsFieldMoves"].reward
        env = ObjectRewardRequiredEventsMapIdsFieldMoves(cfg.env, reward_cfg)
        return OnResetExplorationWrapper(
            env, OmegaConf.create({"full_reset_frequency": 1, "jitter": 1})
        )

    @staticmethod
    def full_state(env) -> str:
        f = io.BytesIO()
        env.unwrapped.pyboy.save_state(f)
        return h(f.getvalue())

    def rollout(self, init_state, env_index, steps, over=None, swarm_from=None, seed=0):
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        env = self.make_env(init_state, env_index, **(over or {}))
        rng = random.Random(1000 + seed)
        obs, info = env.reset(seed=seed)
        trace = {"reset": [obs_digest(obs), canon_digest(info)], "steps": []}
        for i in range(steps):
            if swarm_from is not None and i == steps // 2:
                env.unwrapped.update_state(swarm_from)
                trace["steps"].append({"swarm": self.full_state(env)})
            a = rng.randrange(7)
            try:
                obs, rew, term, trunc, info = env.step(a)
            except Exception as e:  # identical failures are parity too
                trace["steps"].append({"exc": f"{type(e).__name__}: {e}"})
                break
            rec = {
                "a": a,
                "o": obs_digest(obs),
                "r": float(rew).hex(),
                "t": bool(term),
                "u": bool(trunc),
                "i": canon_digest(info),
            }
            if i % 100 == 99:
                rec["s"] = self.full_state(env)
                rec["agent_stats"] = canon_digest(env.unwrapped.agent_stats(a))
            trace["steps"].append(rec)
            if term or trunc:
                obs, info = env.reset()
                trace["steps"].append({"reset_obs": obs_digest(obs)})
        f = io.BytesIO()
        env.unwrapped.pyboy.save_state(f)
        env.close()
        return trace, f.getvalue()

    @staticmethod
    def poke(raw, rng):
        """Write random but valid values into bag / event flags / party / enemy RAM."""
        from pokemonred_puffer.data.events import REQUIRED_EVENTS, EventFlagsBits
        from pokemonred_puffer.data.items import Items
        from pokemonred_puffer.data.species import Species

        pb = raw.pyboy

        def sym(n):
            return pb.symbol_lookup(n)[1]

        r = rng.random()
        if r < 0.08:  # bag
            items = sorted(Items._value2member_map_)
            n = rng.randrange(0, 21)
            bag = []
            for _ in range(n):
                bag += [rng.choice(items), rng.randrange(1, 100)]
            bag += [0xFF] * (40 - len(bag))
            pb.memory[sym("wBagItems") : sym("wBagItems") + 40] = bag[:40]
            pb.memory[sym("wNumBagItems")] = n
        elif r < 0.14:  # required event flags
            names = [f[0] for f in EventFlagsBits._fields_]
            for _ in range(rng.randrange(1, 4)):
                idx = names.index(rng.choice(sorted(REQUIRED_EVENTS)))
                a = 0xD747 + idx // 8
                pb.memory[a] = pb.memory[a] ^ (1 << (idx % 8))
        elif r < 0.18:  # party: count, species, levels, moves (incl. HMs), hp
            cnt = rng.randrange(1, 7)
            pb.memory[sym("wPartyCount")] = cnt
            for i in range(1, cnt + 1):
                pb.memory[sym(f"wPartyMon{i}Species")] = rng.choice(
                    [
                        0x99,
                        0x09,
                        0xB0,
                        0x24,
                        0x15,
                        0x70,
                        rng.choice(sorted(Species._value2member_map_)),
                    ]
                )
                pb.memory[sym(f"wPartyMon{i}Level")] = rng.randrange(1, 100)
                mv = sym(f"wPartyMon{i}Moves")
                pb.memory[mv : mv + 4] = [
                    rng.choice([0x0F, 0x39, 0x46, 0x13, 0x94, rng.randrange(1, 166)])
                    for _ in range(4)
                ]
                hp = sym(f"wPartyMon{i}HP")
                pb.memory[hp] = rng.randrange(0, 2)
                pb.memory[hp + 1] = rng.randrange(256)
        elif r < 0.20:  # enemy party levels
            pb.memory[sym("wEnemyPartyCount")] = rng.randrange(0, 7)
            for i in range(1, 7):
                pb.memory[sym(f"wEnemyMon{i}Level")] = rng.randrange(1, 100)

    def unit_fuzz(self, state, env_index, n, seed):
        """Poke random RAM, then call the env's helpers directly (no emulation) and digest results."""
        import numpy as np

        random.seed(seed)  # self-contained: independent of which scenarios ran before
        np.random.seed(seed)
        rng = random.Random(seed)
        from pokemonred_puffer.data.events import EventFlags
        from pokemonred_puffer.data.flags import Flags
        from pokemonred_puffer.data.missable_objects import MissableFlags
        from pokemonred_puffer.data.party import PartyMons
        from pokemonred_puffer.data.tm_hm import TmHmMoves

        env = self.make_env(state, env_index)
        env.reset(seed=0)
        raw = env.unwrapped
        pb = raw.pyboy
        base = io.BytesIO()
        pb.save_state(base)
        base = base.getvalue()
        maps = [0, 1, 12, 51, 54, 59, 65, 156, 217, 218, 219, 220, 221, 222, 223, 225, 226]
        maps += [159, 161, 162, 108, 82]
        recs = {"steps": [], "reset": ""}
        for i in range(n):
            if i % 10 == 0:
                pb.load_state(io.BytesIO(base))
            for _ in range(3):
                self.poke(raw, rng)
            if rng.random() < 0.3:
                for name in [
                    "wXCoord",
                    "wYCoord",
                    "wCurMap",
                    "wIsInBattle",
                    "wMovementFlags",
                    "wSSAnne2FCurScript",
                    "wObtainedBadges",
                    "wWalkBikeSurfState",
                    "wLastBlackoutMap",
                ]:
                    if rng.random() < 0.3:
                        v = rng.choice(maps) if name == "wCurMap" else rng.randrange(0, 256)
                        pb.memory[pb.symbol_lookup(name)[1]] = v
            out = []
            try:
                raw.events = EventFlags(pb)
                raw.missables = MissableFlags(pb)
                raw.flags = Flags(pb)
                raw.party = PartyMons(pb)
                raw.update_pokedex()
                raw.update_tm_hm_obtained_move_ids()
                out.append(raw.update_max_op_level())
                out.append(raw.read_hp_fraction())
                raw.update_health()
                out += [raw.check_if_party_has_hm(m.value) for m in TmHmMoves]
                out.append(raw.party_has_cut_capable_mon())
                out.append(sorted(raw.get_required_events()))
                out.append(sorted(raw.get_required_items()))
                out.append(raw.get_hm_count())
                out.append(
                    [
                        raw.scale_map_id(m)
                        for m in [0, 1, 12, 51, 54, 59, 65, 156, 159, 161, 162, 108, 82]
                        + [raw.read_m("wCurMap")]
                    ]
                )
                raw.update_seen_coords()
                raw.update_safari_zone()
                raw.update_map_progress()
                rew = raw.get_game_state_reward()
                out.append(rew)
                out.append(sum(v for v in rew.values()))
                out.append(raw.update_reward())
                out.append(obs_digest(raw._get_obs()))
                out.append(raw.agent_stats(0) if i % 15 == 0 else None)
            except Exception as e:
                out.append(f"EXC {type(e).__name__}: {e}")
            recs["steps"].append({"u": canon_digest(out)})
        env.close()
        return recs

    def run(self, steps, group="all", log=print):
        out = {}
        if group in ("all", "rollouts"):
            swarm_state = None
            for s in self.states:
                t = time.perf_counter()
                out[s], final_state = self.rollout(s, self.states.index(s), steps)
                if s == "cut":
                    swarm_state = final_state
                log(f"  {s}: {time.perf_counter() - t:.1f}s")
            n = len(self.states)
            out["short_eps"], _ = self.rollout(
                "Bulbasaur", n, steps, over={"max_steps": 150}, seed=7
            )
            out["swarm"], _ = self.rollout("mtmoon", n + 1, steps, swarm_from=swarm_state, seed=3)
        if group in ("all", "unitfuzz"):
            for k, s in enumerate(self.states):
                env_index = len(self.states) + 2 + k
                out[f"unitfuzz_{s}"] = self.unit_fuzz(s, env_index, UNIT_FUZZ_ROUNDS, seed=900 + k)
        return out


def compact(trace):
    return {
        sc: {
            "reset": record_digest(t["reset"]),
            "steps": [record_digest(rec) for rec in t["steps"]],
        }
        for sc, t in trace.items()
    }


def _explain(g, c):
    """Which fields of a full-format record differ."""
    keys = sorted(k for k in set(g) | set(c) if g.get(k) != c.get(k))
    detail = []
    for k in keys:
        if k == "o" and "o" in g and "o" in c:
            detail.append("obs:" + ",".join(kk for kk in g["o"] if g["o"][kk] != c["o"].get(kk)))
        elif k == "r" and "r" in g and "r" in c:
            detail.append(f"reward {float.fromhex(g['r'])!r} vs {float.fromhex(c['r'])!r}")
        else:
            detail.append(k)
    return "; ".join(detail)


def first_diffs(gold, cand):
    """One message per scenario that diverges from the golden (empty list on parity)."""
    full = gold.get("_meta", {}).get("format") == "full"
    if not full:
        cand = compact(cand)
    bad = []
    for sc, g in gold.items():
        if sc == "_meta" or sc not in cand:
            continue
        c = cand[sc]
        if g["reset"] != c["reset"]:
            bad.append(f"{sc}: reset differs")
            continue
        for i, (gs, cs) in enumerate(zip(g["steps"], c["steps"])):
            if gs != cs:
                why = f": {_explain(gs, cs)}" if full else ""
                bad.append(f"{sc}: record {i} differs{why}")
                break
        else:
            if len(g["steps"]) != len(c["steps"]):
                bad.append(f"{sc}: {len(c['steps'])} records, golden has {len(g['steps'])}")
    missing = [sc for sc in gold if sc != "_meta" and sc not in cand]
    return bad, missing


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def save(obj, path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt") as f:
        json.dump(obj, f, separators=(",", ":"))


def _versions():
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": sys.version.split()[0], "platform": sys.platform}
    for pkg in ("numpy", "pyboy", "scikit-image", "gymnasium", "pufferlib"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("mode", choices=["record", "check"])
    p.add_argument("path")
    p.add_argument("--repo", default=THIS_REPO)
    p.add_argument("--emulator", choices=["fastboy", "pyboy"])
    p.add_argument("--group", choices=["all", "rollouts", "unitfuzz"], default="all")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--full", action="store_true")
    args = p.parse_args(argv)

    rom = os.environ.get("POKEMON_RED_ROM")
    if not rom:
        sys.exit("set POKEMON_RED_ROM to the Pokemon Red ROM path")
    rom = os.path.abspath(rom)
    path = os.path.abspath(args.path)
    repo = os.path.abspath(args.repo)

    gold = load(path) if args.mode == "check" else None
    steps = args.steps or (gold or {}).get("_meta", {}).get("steps", DEFAULT_STEPS)

    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    t0 = time.perf_counter()
    trace = Runner(repo, rom, args.emulator).run(steps, args.group, log=log)
    n = sum(len(v["steps"]) for v in trace.values())
    log(f"{len(trace)} scenarios, {n} records in {time.perf_counter() - t0:.1f}s")

    if args.mode == "record":
        meta = {"format": "full" if args.full else "compact", "steps": steps, **_versions()}
        save({"_meta": meta, **(trace if args.full else compact(trace))}, path)
        print(f"recorded {path}")
        return 0

    bad, missing = first_diffs(gold, trace)
    expected = {
        "all": lambda sc: True,
        "rollouts": lambda sc: not sc.startswith("unitfuzz_"),
        "unitfuzz": lambda sc: sc.startswith("unitfuzz_"),
    }[args.group]
    missing = [sc for sc in missing if expected(sc)]
    extra = [sc for sc in trace if sc not in gold]
    if missing:
        bad.append(f"scenarios in golden but not run: {missing}")
    if extra:
        bad.append(f"scenarios run but not in golden: {extra}")
    if bad:
        print("PARITY FAIL")
        print("\n".join(bad))
        return 1
    print(f"PARITY OK ({len(trace)} scenarios, {n} records)")
    return 0


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":  # agent_stats depends on set iteration order
        env = dict(os.environ, PYTHONHASHSEED="0")
        os.execve(sys.executable, [sys.executable, os.path.abspath(__file__), *sys.argv[1:]], env)
    sys.exit(main())
