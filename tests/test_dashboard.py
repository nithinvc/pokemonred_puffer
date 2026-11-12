from pokemonred_puffer.dashboard import CHAPTERS, Dashboard, section_for


def info(env_id, required_count, events=(), reward=None, coord=0):
    return {
        "env_ids": env_id,
        "stats": {
            "required_count": required_count,
            "max_map_progress": 1,
            "coord": coord,
            "badge": 0,
        },
        "events": {e: True for e in events} | {"EVENT_BEAT_BROCK": False},
        "required_items": {},
        "reward": reward or {},
        "reward_sum": sum((reward or {}).values()),
    }


def test_section_mapping():
    assert section_for("stats/coord") == "exploration/coord"
    assert section_for("stats/menu/start_menu") == "behavior/menu/start_menu"
    assert section_for("stats/exploration/overworld") == "exploration/tileset_share/overworld"
    assert section_for("stats/valid_cut_coords") == "debug/valid_cut_coords"
    assert section_for("stats/levels") is None
    assert section_for("reward_sum") == "reward/total"
    assert section_for("species") is None


def test_aggregates_over_latest_snapshot_of_every_env():
    d = Dashboard()
    d.ingest(info(1, 2, coord=10), global_step=100)
    d.ingest(info(2, 4, coord=30), global_step=200)
    # env 1 reports again; env 2's older snapshot still counts
    d.ingest(info(1, 3, coord=20), global_step=300)
    out = d.epoch_stats()
    assert out["overview/envs_reporting"] == 2
    assert out["exploration/coord"] == 25
    assert out["overview/required_count"] == 3.5
    assert out["overview/required_count_max"] == 4


def test_first_reached_and_story_fractions():
    d = Dashboard()
    d.ingest(info(1, 1, events=["EVENT_GOT_STARTER"]), global_step=50)
    d.ingest(info(2, 0), global_step=60)
    d.ingest(info(2, 1, events=["EVENT_GOT_STARTER"]), global_step=90)
    assert d.first_reached["EVENT_GOT_STARTER"]["step"] == 50
    assert d.first_reached["EVENT_GOT_STARTER"]["env"] == 1
    assert "EVENT_BEAT_BROCK" not in d.first_reached
    out = d.epoch_stats()
    assert out["story/EVENT_GOT_STARTER"] == 1.0
    assert "story/EVENT_BEAT_BROCK" not in out  # only milestones some env has reached
    chapter = CHAPTERS[0]
    assert out[f"chapters/00 {chapter[0]}"] == 1 / len(chapter[1])
    assert [r[5] for r in d.timeline] == ["EVENT_GOT_STARTER"]


def test_event_rewards_fold_into_one_series():
    d = Dashboard()
    d.ingest(info(1, 0, reward={"EVENT_GOT_STARTER": 1.0, "HM_01": 2.0, "coord": 0.5}), 10)
    out = d.epoch_stats()
    assert out["reward/required_events_items"] == 3.0
    assert out["reward/coord"] == 0.5
    assert "reward/HM_01" not in out


def test_marks_count_and_land_on_timeline():
    d = Dashboard()
    d.ingest(info(1, 0), 10)
    d.mark("swarm_migration", "48 envs -> 9 required", 1000)
    d.mark("swarm_migration", "48 envs -> 10 required", 2000)
    assert d.epoch_stats()["overview/swarm_migrations"] == 2
    assert [r[2] for r in d.timeline] == ["swarm_migration", "swarm_migration"]
