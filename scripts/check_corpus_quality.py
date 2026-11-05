"""Quality checks over the pulled transcript corpus.

Checks (per bead cdgl-575):
- every video_id in metadata/titles_all.json has a transcript
- no empty transcripts
- no repetition loops (same segment text repeated >30x)
- no >120s gaps between consecutive segments
- wpm sanity (120-230)
- last segment end within 5% of the lecture duration
"""

import glob
import json
import re
from collections import Counter

CORPUS = "data/corpus/transcription"
TITLES = "metadata/titles_all.json"


def main() -> None:
    titles = json.load(open(TITLES))
    files = sorted(glob.glob(f"{CORPUS}/*.json"))
    by_vid = {}
    for f in files:
        # filename: <date>·<video_id>·transcription.json
        vid = f.split("·")[1]
        by_vid[vid] = f

    missing = sorted(set(titles) - set(by_vid))
    extra = sorted(set(by_vid) - set(titles))
    print(f"transcripts: {len(files)}, titles: {len(titles)}")
    print(f"missing transcripts for video_ids: {len(missing)}")
    for v in missing:
        print(f"  MISSING {v} {titles[v]['title'][:70]}")
    if extra:
        print(f"extra transcripts not in titles: {len(extra)}")
        for v in extra:
            print(f"  EXTRA {v}")

    issues = {k: [] for k in ("empty", "repetition", "gap", "wpm", "tail")}
    for vid, f in sorted(by_vid.items()):
        d = json.load(open(f))
        segs = d.get("segments", [])
        meta = titles.get(vid, {})
        label = f"{vid} {meta.get('date', '?')} {meta.get('title', '')[:60]}"

        words = len(d.get("text", "").split())
        dur = meta.get("duration")
        if not segs or words == 0:
            issues["empty"].append(label)
            continue

        counts = Counter(re.sub(r"\s+", " ", s["text"].strip()) for s in segs)
        worst, n = counts.most_common(1)[0]
        if n > 30:
            issues["repetition"].append(f"{label} | x{n}: {worst[:80]!r}")

        gaps = []
        prev_end = segs[0]["start"]
        for s in segs:
            gap = s["start"] - prev_end
            if gap > 120:
                gaps.append((prev_end, s["start"]))
            prev_end = max(prev_end, s["end"])
        if gaps:
            gtxt = ", ".join(f"{a:.0f}s-{b:.0f}s" for a, b in gaps[:3])
            issues["gap"].append(f"{label} | {len(gaps)} gap(s): {gtxt}")

        if dur:
            minutes = segs[-1]["end"] / 60
            wpm = words / minutes if minutes > 0 else 0
            if not (120 <= wpm <= 230):
                issues["wpm"].append(f"{label} | wpm={wpm:.0f} ({words}w/{minutes:.1f}m)")
            tail_gap = dur - segs[-1]["end"]
            if abs(tail_gap) > 0.05 * dur:
                issues["tail"].append(
                    f"{label} | ends {segs[-1]['end']:.0f}s of {dur}s (off {tail_gap:+.0f}s)"
                )

    print()
    for k, rows in issues.items():
        print(f"=== {k}: {len(rows)} ===")
        for r in rows:
            print(f"  {r}")


if __name__ == "__main__":
    main()
