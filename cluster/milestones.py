"""Human-readable milestone report for a training run.

Runs logged with train.wandb_layout: sections already carry this in wandb
(first_reached/* summary columns, overview/timeline table); this script is for
older runs.

Reads the trainer's state archive (archive_states: True), where each sub-folder
is one distinct set of required events/items some env reached, holding a
desc.txt (the set) and .state files. The earliest .state mtime in each folder
is when that set was first reached, so the first folder containing a milestone
gives when any env first achieved it.

Usage:
  python cluster/milestones.py savio:Workspace/pokemonred_puffer/20261109-201439 \
      --wandb-run nithinc/pokemonrl-red-reproduction/gym2-savio-1109-2009 \
      --html milestones.html
  python cluster/milestones.py path/to/local/archive

The archive argument is a local directory or HOST:PATH (read over ssh).
--wandb-run is optional: it adds agent steps and run-relative times.
"""

import argparse
import ast
import bisect
import html
import os
import subprocess
import sys
from datetime import datetime, timezone

# Story order lives in the package so training-time wandb logging uses the same list.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pokemonred_puffer.dashboard import CHAPTERS  # noqa: E402

# Prints "<first_mtime>|<state_count>|<desc>" per archive folder. Folder names
# can start with '-' (negative hash), hence ./ and --.
REMOTE_SCAN = r"""cd "$1" && for d in ./*/; do
  f=$(ls -tr -- "$d"*.state 2>/dev/null | head -1); [ -n "$f" ] || continue
  echo "$(stat -c %Y -- "$f")|$(cat -- "$d"desc.txt)"
done"""


def scan_archive(archive: str) -> list[tuple[float, frozenset[str]]]:
    if ":" in archive and not os.path.exists(archive):
        host, path = archive.split(":", 1)
        out = subprocess.run(
            ["ssh", host, "bash", "-s", "--", path],
            input=REMOTE_SCAN, capture_output=True, text=True, check=True,
        ).stdout
    else:
        out = subprocess.run(
            ["bash", "-s", "--", archive], input=REMOTE_SCAN,
            capture_output=True, text=True, check=True,
        ).stdout
    sets = []
    for line in out.splitlines():
        ts, desc = line.split("|", 1)
        sets.append((float(ts), frozenset(ast.literal_eval(desc))))
    return sorted(sets, key=lambda s: s[0])


def wandb_lookup(run_path: str):
    """Returns (run_start_epoch, fn: epoch -> agent steps or None, summary dict)."""
    import wandb

    run = wandb.Api(timeout=120).run(run_path)
    rows = [
        (r["_timestamp"], r["Overview/agent_steps"])
        for r in run.scan_history(keys=["_timestamp", "Overview/agent_steps"], page_size=5000)
    ]
    rows.sort()
    ts = [r[0] for r in rows]
    start = run.summary.get("_timestamp", ts[-1]) - run.summary.get("_runtime", 0)

    def steps_at(t):
        i = bisect.bisect_left(ts, t)
        return rows[min(i, len(rows) - 1)][1] if rows else None

    summary = {
        "state": run.state,
        "steps": run.summary.get("Overview/agent_steps"),
        "runtime": run.summary.get("_runtime"),
    }
    return start, steps_at, summary


def fmt_dur(sec: float) -> str:
    sec = max(sec, 0)
    h, m = divmod(int(sec) // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def build(sets, start, steps_at):
    first: dict[str, float] = {}
    for ts, s in sets:
        for k in s:
            first.setdefault(k, ts)
    known = {k for _, items in CHAPTERS for k, _ in items}
    chapters = [
        (name, [(k, label, first.get(k)) for k, label in items]) for name, items in CHAPTERS
    ]
    extra = sorted(set(first) - known)
    if extra:
        chapters.append(("Other", [(k, k, first[k]) for k in extra]))
    rows = []
    for name, items in chapters:
        out = []
        for k, label, t in items:
            out.append({
                "key": k, "label": label, "done": t is not None,
                "elapsed": None if t is None else t - start,
                "steps": None if (t is None or steps_at is None) else steps_at(t),
            })
        rows.append((name, out))
    return rows


def print_text(rows, now_elapsed, summary):
    done = sum(r["done"] for _, items in rows for r in items)
    total = sum(len(items) for _, items in rows)
    hdr = f"Milestones: {done}/{total}"
    if summary:
        hdr += f"   run {summary['state']}, {summary['steps'] / 1e6:.0f}M steps, {fmt_dur(summary['runtime'])}"
    print(hdr)
    last = max((r["elapsed"] for _, it in rows for r in it if r["done"]), default=0)
    print(f"Last new milestone {fmt_dur(now_elapsed - last)} ago\n")
    for name, items in rows:
        n = sum(r["done"] for r in items)
        print(f"{name}  [{n}/{len(items)}]")
        if n == 0:
            continue
        for r in items:
            mark = "x" if r["done"] else " "
            when = ""
            if r["done"]:
                when = fmt_dur(r["elapsed"])
                if r["steps"] is not None:
                    when += f"  {r['steps'] / 1e6:6.1f}M"
            print(f"  [{mark}] {r['label']:<42} {when}")
        print()


def write_html(rows, path, title, now_elapsed, summary):
    done = sum(r["done"] for _, items in rows for r in items)
    total = sum(len(items) for _, items in rows)
    last = max((r["elapsed"] for _, it in rows for r in it if r["done"]), default=0)
    sub = f"{done} of {total} milestones"
    if summary:
        sub += f" · {summary['steps'] / 1e6:.0f}M steps · {fmt_dur(summary['runtime'])} · {summary['state']}"
    sub += f" · last new milestone {fmt_dur(now_elapsed - last)} ago"
    secs = []
    for name, items in rows:
        n = sum(r["done"] for r in items)
        state = "done" if n == len(items) else ("part" if n else "todo")
        lis = []
        for r in items:
            when = ""
            if r["done"]:
                when = fmt_dur(r["elapsed"])
                if r["steps"] is not None:
                    m = r["steps"] / 1e6
                    when += f" · {m:.1f}M" if m < 10 else f" · {m:.0f}M"
            lis.append(
                f'<li class="{"y" if r["done"] else "n"}"><span class="dot"></span>'
                f'<span class="lbl" title="{html.escape(r["key"])}">{html.escape(r["label"])}</span>'
                f'<span class="when">{when}</span></li>'
            )
        secs.append(
            f'<section class="{state}"><h2>{html.escape(name)}<span class="cnt">{n}/{len(items)}</span></h2>'
            f'<div class="bar"><i style="width:{100 * n / len(items):.0f}%"></i></div>'
            f'<ul>{"".join(lis)}</ul></section>'
        )
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Run Milestones</title><style>
:root{{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1f;--mut:#6e6e73;--line:#e3e3e0;--ok:#2f8f5b;--part:#c98a1b;--todo:#b8b8b3}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#141415;--card:#1d1d1f;--fg:#ececec;--mut:#9a9aa0;--line:#2e2e31;--ok:#4cc27f;--part:#e0a23a;--todo:#55555a}}}}
:root[data-theme="dark"]{{--bg:#141415;--card:#1d1d1f;--fg:#ececec;--mut:#9a9aa0;--line:#2e2e31;--ok:#4cc27f;--part:#e0a23a;--todo:#55555a}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,sans-serif;padding:24px 16px}}
.wrap{{max-width:980px;margin:0 auto}}h1{{font-size:22px;margin:0 0 4px}}.sub{{color:var(--mut);margin-bottom:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:12px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}}
section.todo{{opacity:.55}}h2{{font-size:14px;margin:0;display:flex;justify-content:space-between}}
.cnt{{color:var(--mut);font-weight:500;font-variant-numeric:tabular-nums}}
.bar{{height:4px;background:var(--line);border-radius:2px;margin:8px 0 6px;overflow:hidden}}
.bar i{{display:block;height:100%;background:var(--ok)}}section.part .bar i{{background:var(--part)}}
ul{{list-style:none;margin:0;padding:0}}li{{display:flex;align-items:center;gap:8px;padding:3px 0}}
.dot{{width:8px;height:8px;border-radius:50%;flex:none;background:var(--todo)}}li.y .dot{{background:var(--ok)}}
li.n .lbl{{color:var(--mut)}}.lbl{{flex:1}}.when{{color:var(--mut);font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}}
</style></head><body><div class="wrap"><h1>{html.escape(title)}</h1><div class="sub">{html.escape(sub)}</div>
<div class="grid">{"".join(secs)}</div>
<p class="sub" style="margin-top:16px">Time and steps show when any env first reached each milestone (from the state archive). Hover a label for its raw key.</p>
</div></body></html>"""
    with open(path, "w") as f:
        f.write(doc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", help="archive dir, local or HOST:PATH")
    ap.add_argument("--wandb-run", help="entity/project/run_id for steps and run start")
    ap.add_argument("--html", help="also write an HTML report here")
    args = ap.parse_args()

    sets = scan_archive(args.archive)
    if not sets:
        raise SystemExit("no archived states found")
    start, steps_at, summary = sets[0][0], None, None
    if args.wandb_run:
        start, steps_at, summary = wandb_lookup(args.wandb_run)
    now_elapsed = datetime.now(timezone.utc).timestamp() - start
    rows = build(sets, start, steps_at)
    print_text(rows, now_elapsed, summary)
    if args.html:
        name = args.wandb_run.rsplit("/", 1)[-1] if args.wandb_run else os.path.basename(args.archive)
        write_html(rows, args.html, name, now_elapsed, summary)
        print(f"wrote {args.html}")


if __name__ == "__main__":
    main()
