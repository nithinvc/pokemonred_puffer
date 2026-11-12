"""Create (or update) the saved wandb workspace view for runs using
train.wandb_layout: sections.

Usage (repo root):
  python cluster/wandb_dashboard.py --entity nithinc --project pokemonrl-red-dashboard-demo
  python cluster/wandb_dashboard.py ... --view-url URL  # update an existing view in place

Needs `pip install wandb-workspaces` (public preview API), or `pip install -e .[dashboard]`.
"""

import argparse
import os
import sys

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pokemonred_puffer.dashboard import CHAPTERS, chapter_slug  # noqa: E402

X = "overview/agent_steps"


def line(title, y=None, regex=None, **kw):
    kw.setdefault("x", X)
    if regex is not None:
        return wr.LinePlot(title=title, metric_regex=regex, **kw)
    return wr.LinePlot(title=title, y=y if isinstance(y, list) else [y], **kw)


def mean_max(title, key):
    return line(
        title, [key, f"{key}_max"], line_titles={key: "mean over envs", f"{key}_max": "best env"}
    )


def overview_panels():
    return [
        line("Story progress (mean share of milestones per env)", "overview/story_progress"),
        line("Milestones reached by any env", "overview/milestones_reached"),
        mean_max("Required events + items", "overview/required_count"),
        mean_max("Badges", "overview/badges"),
        mean_max("Map progress", "overview/max_map_progress"),
        line("Reward per env (game-state total)", "reward/total"),
        line("Required count across envs (histogram)", "overview/required_count_dist"),
        line("Swarm migrations", "overview/swarm_migrations"),
        line("Agent steps / s", "overview/SPS"),
        wr.WeavePanelSummaryTable(table_name="overview/timeline"),
    ]


def story_sections():
    chapter_keys = [f"chapters/{chapter_slug(i, c)}" for i, (c, _) in enumerate(CHAPTERS)]
    sections = [
        ws.Section(
            name="Story: chapter completion",
            is_open=True,
            panels=[
                line(
                    "Chapter completion (mean share of a chapter's milestones per env)",
                    chapter_keys,
                    line_titles={k: k.split("/", 1)[1] for k in chapter_keys},
                )
            ],
        )
    ]
    panels = []
    for _, (chapter, items) in enumerate(CHAPTERS):
        keys = [f"story/{k}" for k, _ in items]
        panels.append(
            line(
                f"{chapter}: share of envs that have each",
                keys,
                line_titles={f"story/{k}": label for k, label in items},
                range_y=(0, 1),
            )
        )
    sections.append(
        ws.Section(
            name="Story: milestones by chapter",
            is_open=True,
            panels=panels,
            layout_settings=ws.SectionLayoutSettings(columns=2, rows=3),
        )
    )
    return sections


def workspace(entity, project, name):
    sections = [
        ws.Section(name="Overview", is_open=True, panels=overview_panels()),
        *story_sections(),
        ws.Section(
            name="Reward",
            is_open=True,
            panels=[
                line(
                    "Reward components (stacked)",
                    regex="reward/(?!total).*",
                    plot_type="stacked-area",
                ),
                line("Total", "reward/total"),
            ],
        ),
        ws.Section(
            name="Exploration",
            is_open=True,
            panels=[
                wr.MediaBrowser(title="Aggregate exploration map", media_keys=["exploration/map"]),
                line("Coords seen", "exploration/coord"),
                line("Warps / maps", ["exploration/warps", "exploration/map_id"]),
                line(
                    "NPCs, signs, hidden objects",
                    ["exploration/npc", "exploration/sign", "exploration/hidden_obj"],
                ),
                line(
                    "Where time is spent (tileset share)",
                    regex="exploration/tileset_share/.*",
                    plot_type="pct-area",
                ),
                line("Cut tiles", "exploration/cut_tiles"),
            ],
        ),
        ws.Section(
            name="Party",
            is_open=True,
            panels=[
                line("Party level sum", "party/levels_sum"),
                line("Party size", "party/party_count"),
                line("Pokédex", ["party/caught_pokemon", "party/seen_pokemon"]),
                line("Max opponent level", "party/opponent_level"),
                line("HP fraction", "party/hp"),
                line(
                    "HMs taught", ["party/taught_cut", "party/taught_surf", "party/taught_strength"]
                ),
                wr.WeavePanelSummaryTable(table_name="party/roster"),
            ],
        ),
        ws.Section(
            name="Behavior",
            panels=[
                line("Deaths / blackouts", ["behavior/deaths", "behavior/blackout_count"]),
                line(
                    "Pokécenter visits / heals", ["behavior/pokecenter", "behavior/pokecenter_heal"]
                ),
                line("Menus opened", regex="behavior/menu/.*"),
                line("Unique A presses", "behavior/a_press"),
                line("Share in battle", "behavior/in_battle"),
                line("Episode resets", "behavior/reset_count"),
            ],
        ),
        ws.Section(
            name="Items", panels=[line("Useful items held (share of envs)", regex="items/.*")]
        ),
        ws.Section(
            name="Learning",
            panels=[
                line(k, f"losses/{k}")
                for k in [
                    "policy_loss",
                    "value_loss",
                    "entropy",
                    "approx_kl",
                    "clipfrac",
                    "explained_variance",
                ]
            ]
            + [line("Learning rate", "overview/learning_rate")],
        ),
        ws.Section(name="Performance", panels=[line("Time split", regex="performance/.*")]),
        ws.Section(name="Debug", panels=[line("Debug counters", regex="debug/.*")]),
    ]
    return ws.Workspace(
        entity=entity,
        project=project,
        name=name,
        sections=sections,
        settings=ws.WorkspaceSettings(x_axis=X, max_runs=20),
        runset_settings=ws.RunsetSettings(
            pinned_columns=[
                "summary:overview/milestones_reached",
                "summary:overview/badges_max",
                "summary:overview/story_progress",
                "summary:first_reached/EVENT_BEAT_BROCK",
                "summary:first_reached/EVENT_BEAT_MISTY",
                "config:cleanrl.wandb_layout",
            ]
        ),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", default="Run dashboard")
    ap.add_argument("--view-url", help="existing saved view to overwrite")
    args = ap.parse_args()

    view = workspace(args.entity, args.project, args.name)
    if args.view_url:
        old = ws.Workspace.from_url(args.view_url)
        old.sections, old.settings, old.runset_settings = (
            view.sections,
            view.settings,
            view.runset_settings,
        )
        old.save()
        print("updated view:", old.url)
    else:
        view.save()
        print("view:", view.url)


if __name__ == "__main__":
    main()
