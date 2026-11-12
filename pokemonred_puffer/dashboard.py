"""Structured wandb logging for training runs (train.wandb_layout: sections).

The legacy layout logs every key from RedGymEnv.agent_stats under environment/,
averaged over only the envs whose info arrived that epoch. This module instead:

- keeps each env's latest info snapshot and aggregates over all envs, so curves
  don't jump when a different subset of envs happens to report;
- maps keys into wandb sections (overview/, story/, chapters/, reward/,
  exploration/, party/, behavior/, items/, debug/);
- logs story milestones as the fraction of envs that have reached each one, and
  records when any env first reached it (run summary + timeline table), which
  replaces cluster/milestones.py's archive scan;
- records swarm migrations on the same timeline.
"""

import math
import time
from collections import Counter

import numpy as np

# Story order: (chapter, [(key, label), ...]). Keys are REQUIRED_EVENTS,
# the 4 synthetic events from get_required_events, and REQUIRED_ITEMS names.
CHAPTERS = [
    (
        "Pallet & Viridian",
        [
            ("EVENT_FOLLOWED_OAK_INTO_LAB", "Followed Oak into the lab"),
            ("EVENT_FOLLOWED_OAK_INTO_LAB_2", "Followed Oak into the lab (2)"),
            ("EVENT_OAK_ASKED_TO_CHOOSE_MON", "Oak asked to choose a starter"),
            ("EVENT_GOT_STARTER", "Got starter"),
            ("EVENT_BATTLED_RIVAL_IN_OAKS_LAB", "Rival battle in Oak's lab"),
            ("EVENT_GOT_OAKS_PARCEL", "Picked up Oak's Parcel"),
            ("OAKS_PARCEL", "Item: Oak's Parcel (leaves bag on delivery)"),
            ("EVENT_OAK_GOT_PARCEL", "Delivered the parcel"),
            ("EVENT_GOT_POKEDEX", "Got the Pokédex"),
        ],
    ),
    (
        "Pewter & Mt. Moon",
        [
            ("EVENT_BEAT_BROCK", "Badge 1: Brock"),
            ("EVENT_BEAT_MT_MOON_EXIT_SUPER_NERD", "Beat Mt. Moon Super Nerd (fossil)"),
        ],
    ),
    (
        "Cerulean",
        [
            ("EVENT_BEAT_CERULEAN_RIVAL", "Rival on Nugget Bridge"),
            ("EVENT_BEAT_MISTY", "Badge 2: Misty"),
            ("EVENT_MET_BILL", "Met Bill"),
            ("EVENT_BILL_SAID_USE_CELL_SEPARATOR", "Bill: use the Cell Separator"),
            ("EVENT_USED_CELL_SEPARATOR_ON_BILL", "Used the Cell Separator"),
            ("EVENT_MET_BILL_2", "Met Bill (restored)"),
            ("EVENT_GOT_SS_TICKET", "Got the S.S. Ticket"),
            ("S_S_TICKET", "Item: S.S. Ticket"),
            ("EVENT_LEFT_BILLS_HOUSE_AFTER_HELPING", "Left Bill's house"),
            ("EVENT_BEAT_CERULEAN_ROCKET_THIEF", "Beat the Cerulean Rocket thief"),
        ],
    ),
    (
        "Vermilion",
        [
            ("EVENT_GOT_BIKE_VOUCHER", "Got the Bike Voucher (Fan Club)"),
            ("rival3", "Rival on the S.S. Anne"),
            ("EVENT_RUBBED_CAPTAINS_BACK", "Rubbed the Captain's back"),
            ("EVENT_GOT_HM01", "Got HM01 Cut"),
            ("HM_01", "Item: HM01"),
            ("EVENT_GOT_BICYCLE", "Got the Bicycle"),
            ("EVENT_1ST_LOCK_OPENED", "Vermilion Gym: 1st lock"),
            ("EVENT_2ND_LOCK_OPENED", "Vermilion Gym: 2nd lock"),
            ("EVENT_BEAT_LT_SURGE", "Badge 3: Lt. Surge"),
        ],
    ),
    (
        "Route 9 & Rock Tunnel",
        [
            ("EVENT_BEAT_ROUTE_9_TRAINER_0", "Route 9 lass"),
            ("EVENT_BEAT_ROCK_TUNNEL_1_TRAINER_3", "Rock Tunnel 1F Super Nerd"),
            ("EVENT_BEAT_ROCK_TUNNEL_2_TRAINER_7", "Rock Tunnel B1F Super Nerd"),
            ("EVENT_BEAT_ROCK_TUNNEL_2_TRAINER_5", "Rock Tunnel B1F trainer"),
            ("EVENT_BEAT_ROCK_TUNNEL_2_TRAINER_1", "Rock Tunnel B1F Graveler trainer"),
            ("EVENT_BEAT_ROCK_TUNNEL_1_TRAINER_5", "Rock Tunnel 1F lass (exit)"),
        ],
    ),
    (
        "Lavender (first visit)",
        [
            ("EVENT_BEAT_POKEMON_TOWER_RIVAL", "Rival in Pokémon Tower"),
        ],
    ),
    (
        "Celadon",
        [
            ("game_corner_rocket", "Beat the Game Corner Rocket"),
            ("EVENT_FOUND_ROCKET_HIDEOUT", "Found the Rocket Hideout"),
            ("EVENT_ROCKET_HIDEOUT_4_DOOR_UNLOCKED", "Hideout B4F door unlocked"),
            ("EVENT_ROCKET_DROPPED_LIFT_KEY", "Rocket dropped the Lift Key"),
            ("LIFT_KEY", "Item: Lift Key"),
            ("EVENT_BEAT_ROCKET_HIDEOUT_GIOVANNI", "Beat Giovanni (Hideout)"),
            ("SILPH_SCOPE", "Item: Silph Scope"),
            ("EVENT_BEAT_ERIKA", "Badge 4: Erika"),
        ],
    ),
    (
        "Lavender (Pokémon Tower)",
        [
            ("EVENT_BEAT_GHOST_MAROWAK", "Beat the ghost Marowak"),
            ("EVENT_RESCUED_MR_FUJI", "Rescued Mr. Fuji"),
            ("EVENT_RESCUED_MR_FUJI_2", "Rescued Mr. Fuji (2)"),
            ("EVENT_GOT_POKE_FLUTE", "Got the Poké Flute"),
            ("POKE_FLUTE", "Item: Poké Flute"),
        ],
    ),
    (
        "Fuchsia & Safari Zone",
        [
            ("EVENT_BEAT_KOGA", "Badge 5: Koga"),
            ("EVENT_GOT_HM03", "Got HM03 Surf"),
            ("HM_03", "Item: HM03"),
            ("GOLD_TEETH", "Item: Gold Teeth"),
            ("EVENT_GAVE_GOLD_TEETH", "Gave the Gold Teeth to the Warden"),
            ("EVENT_GOT_HM04", "Got HM04 Strength"),
            ("HM_04", "Item: HM04"),
        ],
    ),
    (
        "Saffron",
        [
            ("saffron_guard", "Gave a drink to the Saffron guard"),
            ("CARD_KEY", "Item: Card Key"),
            ("EVENT_BEAT_SILPH_CO_RIVAL", "Rival in Silph Co."),
            ("lapras", "Got Lapras"),
            ("EVENT_BEAT_SILPH_CO_GIOVANNI", "Beat Giovanni (Silph Co.)"),
            ("EVENT_BEAT_SABRINA", "Badge 6: Sabrina"),
        ],
    ),
    (
        "Cinnabar",
        [
            ("SECRET_KEY", "Item: Secret Key"),
            ("EVENT_BEAT_BLAINE", "Badge 7: Blaine"),
        ],
    ),
    (
        "Viridian Gym & Victory Road",
        [
            ("EVENT_BEAT_VIRIDIAN_GYM_GIOVANNI", "Badge 8: Giovanni"),
            ("EVENT_2ND_ROUTE22_RIVAL_BATTLE", "Route 22 rival (2nd) triggered"),
            ("EVENT_BEAT_ROUTE22_RIVAL_2ND_BATTLE", "Beat the Route 22 rival (2nd)"),
            ("EVENT_PASSED_CASCADEBADGE_CHECK", "Route 23: passed the Cascade Badge gate"),
            ("EVENT_PASSED_EARTHBADGE_CHECK", "Route 23: passed the Earth Badge gate"),
        ],
    ),
    (
        "Elite Four",
        [
            ("EVENT_BEAT_LORELEIS_ROOM_TRAINER_0", "Lorelei"),
            ("EVENT_BEAT_BRUNOS_ROOM_TRAINER_0", "Bruno"),
            ("EVENT_BEAT_AGATHAS_ROOM_TRAINER_0", "Agatha"),
            ("EVENT_BEAT_LANCE", "Lance"),
            ("EVENT_BEAT_CHAMPION_RIVAL", "Champion"),
            ("ELITE4_CHAMPION_EVENTS_END", "Hall of Fame"),
        ],
    ),
]

MILESTONES = {
    key: (ci, chapter, label)
    for ci, (chapter, items) in enumerate(CHAPTERS)
    for key, label in items
}


def chapter_slug(index: int, chapter: str) -> str:
    return f"{index:02d} {chapter}"


# Flattened agent_stats keys (pufferlib.utils.unroll_nested_dict) under "stats/".
STATS_SECTIONS = {
    "exploration": {
        "coord",
        "warps",
        "map_id",
        "npc",
        "hidden_obj",
        "sign",
        "max_map_progress",
        "cut_tiles",
    },
    "party": {
        "party_count",
        "levels_sum",
        "hp",
        "caught_pokemon",
        "seen_pokemon",
        "obtained_move_ids",
        "opponent_level",
        "taught_cut",
        "taught_surf",
        "taught_strength",
    },
    "behavior": {
        "a_press",
        "deaths",
        "blackout_count",
        "blackout_check",
        "pokecenter",
        "pokecenter_heal",
        "in_battle",
        "healr",
        "item_count",
        "use_ball_count",
        "reset_count",
        "step",
        "max_steps",
    },
}
# Keys that are already reported elsewhere, are not scalars, or are not meaningful as means.
SKIP_STATS = {"levels", "ptypes", "last_action", "action_hist", "required_count", "event"}


def section_for(key: str) -> str | None:
    """wandb key for one flattened info key, or None to drop it."""
    parts = key.split("/")
    head = parts[0]
    if head == "stats":
        name = "/".join(parts[1:])
        if name in SKIP_STATS:
            return None
        if name == "badge":
            return "overview/badges"
        if name.startswith("badge_"):
            return f"overview/has_{name}"
        if name.startswith("exploration/"):
            return f"exploration/tileset_share/{parts[2]}"
        if name.startswith("menu/"):
            return f"behavior/menu/{parts[2]}"
        if name.startswith("safari_zone/"):
            return f"behavior/safari_steps/{parts[2]}"
        for section, names in STATS_SECTIONS.items():
            if name in names:
                return f"{section}/{name}"
        return f"debug/{name}"
    if head == "reward":
        return f"reward/{'/'.join(parts[1:])}"
    if head == "reward_sum":
        return "reward/total"
    if head == "useful_items":
        return f"items/{parts[1]}"
    return None


def _scalar(v):
    if isinstance(v, (bool, np.bool_)):
        return float(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, np.ndarray) and v.size == 1:
        return _scalar(v.item())
    return None


class Dashboard:
    def __init__(self, wandb_client=None, alerts: bool = False):
        self.wandb = wandb_client
        self.alerts = alerts
        self.start = time.time()
        # env id -> {wandb key: value}, the latest numeric snapshot per env
        self.latest: dict[int, dict[str, float]] = {}
        # env id -> set of milestone keys reached in its latest snapshot
        self.reached: dict[int, set[str]] = {}
        self.first_reached: dict[str, dict] = {}
        self.timeline: list[list] = []
        self.timeline_dirty = False
        self.counters = Counter()

    def minutes(self) -> float:
        return (time.time() - self.start) / 60

    def ingest(self, info: dict, global_step: int):
        """One env info dict (nested, as returned by RedGymEnv.step)."""
        if "env_ids" not in info or "stats" not in info:
            return
        env_id = int(info["env_ids"])
        snap = {}
        for k, v in _flatten(info):
            wk = section_for(k)
            if wk is None:
                continue
            s = _scalar(v)
            if s is None:
                continue
            if wk.startswith("reward/") and wk[7:].isupper():
                # Rewards keyed by event/item name (one per required event or item): one series.
                wk = "reward/required_events_items"
                s += snap.get(wk, 0.0)
            snap[wk] = s
        stats = info["stats"]
        snap["overview/required_count"] = float(stats.get("required_count", 0))
        snap["overview/max_map_progress"] = float(stats.get("max_map_progress", 0))
        self.latest[env_id] = snap

        reached = {k for k, v in info.get("events", {}).items() if v}
        reached |= {k for k, v in info.get("required_items", {}).items() if v}
        self.reached[env_id] = reached
        for key in reached:
            if key not in self.first_reached:
                self._first(key, env_id, global_step)

    def _first(self, key: str, env_id: int, global_step: int):
        ci, chapter, label = MILESTONES.get(key, (len(CHAPTERS), "Other", key))
        minutes = round(self.minutes(), 2)
        self.first_reached[key] = {"step": int(global_step), "minutes": minutes, "env": env_id}
        self.timeline.append(
            [int(global_step), minutes, "milestone", chapter_slug(ci, chapter), label, key, env_id]
        )
        self.timeline_dirty = True
        if self.wandb is not None:
            self.wandb.summary[f"first_reached/{key}"] = int(global_step)
            self.wandb.summary[f"first_reached_minutes/{key}"] = minutes
            self.wandb.summary["overview/milestones_reached"] = len(self.first_reached)
            if self.alerts and label.startswith("Badge"):
                self.wandb.alert(
                    title=label,
                    text=f"{label} first reached by env {env_id} at {global_step:,} agent steps "
                    f"({minutes:.0f} min).",
                )

    def mark(self, kind: str, detail: str, global_step: int):
        """A run event worth seeing next to the curves, e.g. a swarm migration."""
        self.counters[kind] += 1
        self.timeline.append([int(global_step), round(self.minutes(), 2), kind, "", detail, "", -1])
        self.timeline_dirty = True

    def epoch_stats(self) -> dict:
        """Aggregates over every env's latest snapshot. Returns plain floats/wandb objects."""
        import wandb

        out: dict = {}
        if not self.latest:
            return out
        n = len(self.latest)
        keys = set().union(*(s.keys() for s in self.latest.values()))
        for k in keys:
            vals = [s[k] for s in self.latest.values() if k in s]
            out[k] = float(np.mean(vals))
        for k in ("overview/badges", "overview/required_count", "overview/max_map_progress"):
            vals = [s[k] for s in self.latest.values() if k in s]
            if vals:
                out[f"{k}_max"] = float(np.max(vals))
        out["overview/envs_reporting"] = n
        out["overview/milestones_reached"] = len(self.first_reached)
        for kind, c in self.counters.items():
            out[f"overview/{kind}s"] = c
        req = [s.get("overview/required_count", 0) for s in self.latest.values()]
        out["overview/required_count_dist"] = wandb.Histogram(
            req, num_bins=min(64, max(8, int(max(req)) + 1))
        )

        # Story: fraction of envs that have each milestone, and chapter completion.
        reached = list(self.reached.values())
        for ci, (chapter, items) in enumerate(CHAPTERS):
            fracs = []
            for key, _ in items:
                f = sum(key in r for r in reached) / n
                fracs.append(f)
                if key in self.first_reached:
                    out[f"story/{key}"] = f
            out[f"chapters/{chapter_slug(ci, chapter)}"] = float(np.mean(fracs))
        total = sum(len(items) for _, items in CHAPTERS)
        out["overview/story_progress"] = float(
            np.mean([sum(k in MILESTONES for k in r) / total for r in reached])
        )
        return out

    def timeline_table(self):
        import wandb

        self.timeline_dirty = False
        return wandb.Table(
            columns=["agent_step", "minutes", "kind", "chapter", "label", "key", "env_id"],
            data=[list(r) for r in self.timeline],
        )


def _flatten(d, prefix=""):
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from _flatten(v, f"{key}/")
        else:
            yield key, v
