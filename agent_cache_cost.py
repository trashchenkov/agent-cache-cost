"""How much do compaction and cache misses cost in your Claude Code and Codex sessions?

Reads the local session logs (~/.claude/projects, ~/.codex/sessions) and uses only token counters,
timestamps and compaction markers. Message contents are never read or printed.

  python agent_cache_cost.py summary [--since 2026-09-03] [--until 2026-10-27]
  python agent_cache_cost.py list [--tool claude|codex] [--since ...]
  python agent_cache_cost.py session <id-prefix> [--plot session.png]

Token spend is counted in "input-token equivalents": uncached input 1, cache read 0.1, output 5,
cache write 1.25 (5-minute TTL) or 2.0 (1-hour TTL) for Claude; Codex has no cache-write surcharge.
"""
import argparse, glob, json, os, statistics as st, subprocess
from collections import Counter, defaultdict
from datetime import datetime

READ, OUT, W5, W60 = 0.1, 5.0, 1.25, 2.0
MISS_TOKENS = 20_000  # a request that re-processes more than this and reads under half from cache


def ts(s): return datetime.fromisoformat(s.replace("Z", "+00:00"))


def records(path):
    if path.endswith(".zst"):
        text = subprocess.run(["zstd", "-dc", path], capture_output=True, text=True, errors="ignore").stdout.splitlines()
    else:
        text = open(path, encoding="utf-8", errors="ignore")
    for line in text:
        try: r = json.loads(line)
        except ValueError: continue  # a session still being written can end mid-line
        if isinstance(r, dict): yield r


class Call:
    __slots__ = "t ctx cr fresh out cost extra model after_comp".split()


def load_claude():
    """Yields (session id, calls, compaction times). Subagent requests are their own sessions."""
    for path in glob.glob(os.path.expanduser("~/.claude/projects/**/*.jsonl"), recursive=True):
        streams, comps, seen = defaultdict(list), [], set()
        sid = os.path.basename(path)[:-6]
        for r in records(path):
            if r.get("subtype") == "compact_boundary": comps.append(ts(r["timestamp"]))
            m = r.get("message") if isinstance(r.get("message"), dict) else None
            if r.get("type") != "assistant" or not m or not m.get("usage"): continue
            key = r.get("requestId") or m.get("id")
            if key in seen: continue
            seen.add(key)
            u = m["usage"]; w60 = (u.get("cache_creation") or {}).get("ephemeral_1h_input_tokens", 0) or 0
            w5 = (u.get("cache_creation_input_tokens") or 0) - w60
            c = Call(); c.t = ts(r["timestamp"]); c.model = m.get("model")
            inp, c.cr, c.out = u.get("input_tokens") or 0, u.get("cache_read_input_tokens") or 0, u.get("output_tokens") or 0
            c.fresh = inp + w5 + w60; c.ctx = c.fresh + c.cr
            c.cost = inp + READ * c.cr + W5 * w5 + W60 * w60 + OUT * c.out
            c.extra = (W5 - READ) * w5 + (W60 - READ) * w60 + (1 - READ) * inp  # paid above a cache read
            streams[sid + ("/sub" if r.get("isSidechain") else "")].append(c)
        for k, cs in streams.items(): yield "claude", k, cs, (comps if not k.endswith("/sub") else [])


def load_codex():
    files = glob.glob(os.path.expanduser("~/.codex/sessions/**/*.jsonl"), recursive=True)
    plain = set(files)
    files += [z for z in glob.glob(os.path.expanduser("~/.codex/sessions/**/*.jsonl.zst"), recursive=True) if z[:-4] not in plain]
    for path in files:
        recs = list(records(path))
        new_format = any(r.get("type") == "token_usage_record" for r in recs)
        streams, comps, seen, model = defaultdict(list), defaultdict(list), set(), None
        fid = os.path.basename(path).split(".")[0][-36:]
        for r in recs:
            p = r.get("payload") if isinstance(r.get("payload"), dict) else {}
            if r.get("type") == "turn_context": model = p.get("model")
            if r.get("type") == "compacted":
                th = (p.get("latest_token_usage_record") or {}).get("thread_id") or fid
                comps[th].append(ts(r["timestamp"])); continue
            if new_format and r.get("type") == "token_usage_record":
                if p.get("response_id") in seen: continue
                seen.add(p.get("response_id")); u = p.get("usage") or {}; th = p.get("thread_id") or fid
            elif not new_format and p.get("type") == "token_count" and p.get("info"):
                u = p["info"].get("last_token_usage") or {}; th = fid
                key = (u.get("input_tokens"), u.get("cached_input_tokens"), u.get("output_tokens"))
                if key in seen: continue  # the same usage is re-emitted after other events
                seen = {key}
            else: continue
            c = Call(); c.t = ts(r["timestamp"]); c.model = model
            c.ctx, c.cr, c.out = u.get("input_tokens") or 0, u.get("cached_input_tokens") or 0, u.get("output_tokens") or 0
            if not c.ctx: continue
            c.fresh = c.ctx - c.cr
            c.cost = c.fresh + READ * c.cr + OUT * c.out
            c.extra = (1 - READ) * c.fresh
            streams[th].append(c)
        for th, cs in streams.items():
            # old-format compaction records carry no thread id and belong to the file's main thread
            yield "codex", th, cs, comps.get(th, []) + (comps.get(fid, []) if th != fid else [])


def analyse(cs, comp_times):
    """Marks each call's cause if it re-processed the cached prefix; returns compactions found."""
    cs.sort(key=lambda c: c.t)
    for c in cs: c.after_comp = False
    comps = []
    for t in sorted(comp_times):
        before = [c for c in cs if c.t <= t]; after = [c for c in cs if c.t > t]
        # a compaction inside the session: requests before it, and a shorter context after it
        # (Codex also writes compaction records at the start of resumed sessions; those have nothing before)
        if before and after and after[0].ctx < before[-1].ctx:
            after[0].after_comp = True
            comps.append({"t": t, "pre": before[-1].ctx, "post": after[0].ctx, "rebuild": after[0].extra})
    causes = []
    for prev, c in zip([None] + cs, cs):
        if prev is None or c.fresh < MISS_TOKENS or c.cr > 0.5 * c.ctx: causes.append(None); continue
        gap = (c.t - prev.t).total_seconds()
        causes.append("compaction" if c.after_comp else "model switch" if c.model and prev.model and c.model != prev.model
                      else "pause > 1h" if gap > 3600 else "pause 5-60 min" if gap > 300 else "other")
    return causes, comps


def collect(args):
    out = []
    for tool, sid, cs, comp_times in (list(load_claude()) if args.tool in (None, "claude") else []) + \
                                     (list(load_codex()) if args.tool in (None, "codex") else []):
        cs = [c for c in cs if (not args.since or f"{c.t:%Y-%m-%d}" >= args.since) and (not args.until or f"{c.t:%Y-%m-%d}" <= args.until)]
        if not cs: continue
        causes, comps = analyse(cs, comp_times)
        out.append((tool, sid, cs, causes, comps))
    return out


def summary(args):
    for tool in ("claude", "codex"):
        ss = [s for s in collect(args) if s[0] == tool]
        if not ss: continue
        calls = [c for s in ss for c in s[2]]; total = sum(c.cost for c in calls)
        by = Counter(); n = Counter()
        for s in ss:
            for c, k in zip(s[2], s[3]):
                if k: by[k] += c.extra; n[k] += 1
        comps = [x for s in ss for x in s[4]]
        sessions = len({s[1].split("/")[0] for s in ss})
        print(f"\n{tool}: {sessions} sessions, {len(calls)} requests, {len(comps)} compactions, "
              f"{min(c.t for c in calls):%Y-%m-%d}..{max(c.t for c in calls):%Y-%m-%d}")
        print(f"  requests per compaction: {len(calls) / max(len(comps), 1):.0f}")
        if comps:
            print(f"  context before -> after compaction (median): {st.median(x['pre'] for x in comps)/1e3:.0f}K -> "
                  f"{st.median(x['post'] for x in comps)/1e3:.0f}K")
            be = sorted(x["rebuild"] / (READ * (x["pre"] - x["post"])) for x in comps)
            print(f"  rebuild after compaction pays back in (requests): median {st.median(be):.1f}, 90% within {be[int(.9 * len(be))]:.1f}")
        print("  extra paid for re-processing the prefix, share of all spend:")
        for k, v in by.most_common(): print(f"    {k:15} {n[k]:6}  {v / total:6.1%}")


def listing(args):
    rows = []
    for tool, sid, cs, causes, comps in collect(args):
        rows.append((cs[0].t, tool, sid, len(cs), len(comps), causes.count("pause > 1h"), max(c.ctx for c in cs)))
    print(f"{'start':16} {'tool':6} {'session':40} {'requests':>8} {'compact':>7} {'pauses':>6} {'max ctx':>8}")
    for r in sorted(rows):
        print(f"{r[0]:%Y-%m-%d %H:%M} {r[1]:6} {r[2][:40]:40} {r[3]:8} {r[4]:7} {r[5]:6} {r[6]/1e3:7.0f}K")


def session(args):
    found = [s for s in collect(args) if s[1].startswith(args.id)]
    if not found: raise SystemExit(f"no session starting with {args.id}")
    tool, sid, cs, causes, comps = max(found, key=lambda s: len(s[2]))
    total = sum(c.cost for c in cs)
    print(f"{tool} {sid}: {len(cs)} requests, {len(comps)} compactions, spend {total/1e3:.0f}K input-token equivalents")
    for x in comps: print(f"  compaction {x['t']:%m-%d %H:%M}: {x['pre']/1e3:.0f}K -> {x['post']/1e3:.0f}K tokens")
    for k, v in Counter(k for k in causes if k).most_common():
        print(f"  {k:15} {v:4} requests, {sum(c.extra for c, kk in zip(cs, causes) if kk == k) / total:.1%} of the session's spend")
    if args.plot: plot(tool, cs, causes, args.plot)


def plot(tool, cs, causes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, GRAY, INK, MUTED, GRID, BG = "#2a78d6", "#eb6834", "#8f8d86", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
    color = {"pause > 1h": ORANGE, "other": GRAY, "model switch": GRAY, "pause 5-60 min": GRAY}
    plt.rcParams.update({"font.size": 22, "font.family": ["Arial", "DejaVu Sans"]})
    fig, ax = plt.subplots(figsize=(9, 8), dpi=150, facecolor=BG); ax.set_facecolor(BG)
    ys = [c.cost / 1e3 for c in cs]
    ax.bar(range(len(cs)), ys, width=0.9, linewidth=0, color=[color.get(k, BLUE) for k in causes])
    for i, c in enumerate(cs):
        if c.after_comp:
            ax.axvline(i - 0.5, color=INK, linewidth=2, linestyle=(0, (5, 4)))
            ax.text(i + len(cs) * 0.008, max(ys) * 0.98, "компактизация", color=INK, fontsize=18, va="top", rotation=90)
    ax.set_xlim(-1, len(cs))
    ax.set_ylabel("тыс. условных входных токенов", color=MUTED, fontsize=20)
    ax.set_xlabel("запросы по порядку", color=MUTED, fontsize=20)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    for side in ("left", "bottom"): ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=19); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    name = {"claude": "Claude Code", "codex": "Codex"}[tool]
    fig.text(0.04, 0.95, f"Расход токенов на каждый запрос\nв моей сессии {name}", fontsize=28, fontweight="bold", color=INK, va="top")
    for y, col, label in ((0.80, ORANGE, "после паузы больше часа"), (0.755, BLUE, "обычный запрос"),
                          (0.71, GRAY, "кеш сбросился по другой причине")):
        fig.text(0.04, y, "■", color=col, fontsize=24, va="center"); fig.text(0.08, y, label, color=INK, fontsize=22, va="center")
    fig.subplots_adjust(top=0.665, left=0.14, right=0.97, bottom=0.11)
    fig.savefig(path, facecolor=BG); print("saved", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("summary", "list", "session"):
        p = sub.add_parser(name)
        p.add_argument("--tool", choices=["claude", "codex"]); p.add_argument("--since", help="YYYY-MM-DD"); p.add_argument("--until", help="YYYY-MM-DD")
        if name == "session": p.add_argument("id", help="session id or its prefix"); p.add_argument("--plot")
    a = ap.parse_args()
    {"summary": summary, "list": listing, "session": session}[a.cmd](a)
