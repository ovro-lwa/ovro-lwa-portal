"""Print markdown tables (median [min–max], ms) from bench_linked JSONL results."""

from __future__ import annotations

import json
import sys

COLS = [
    ("T_sky_paint", "sky paint"),
    ("T_dynspec_first_paint", "dynspec first paint"),
    ("T_dynspec_complete_paint", "dynspec complete paint"),
    ("T_link_paint", "link paint"),
    ("max_stall", "max stall"),
]


def fmt(stat: dict) -> str:
    return f"{stat['median'] * 1e3:.0f} [{stat['min'] * 1e3:.0f}–{stat['max'] * 1e3:.0f}]"


def main(paths: list[str]) -> None:
    print("| Layout | Approach | Workers | " + " | ".join(c[1] for c in COLS) + " | paints | sky chunks / MB | dyn chunks / MB |")
    print("|" + "---|" * (len(COLS) + 6))
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            s = r["summary"]
            print(
                f"| {r['label']} | {r['approach']} | {r['workers']} | "
                + " | ".join(fmt(s[k]) for k, _ in COLS)
                + f" | {s['n_paints']['median']:.0f}"
                + f" | {r['sky_chunks']} / {r['sky_bytes'] / 1e6:.1f}"
                + f" | {r['dyn_chunks']} / {r['dyn_bytes'] / 1e6:.0f} |"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
