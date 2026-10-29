"""Look for duplicate roads in ``roads_within_5_miles.json`` and remove them.

A "road" is treated as a *named street*: entries that share the same ``name``
(ignoring case and surrounding spaces) are considered duplicates of each other.
The file is already sorted by name (then nearest-first), so the first entry for
a name is kept as the representative.

The report also prints the counts for two stricter definitions so you can see
the numbers either way:

* same name,
* same name **and** same ``maxspeed``,
* identical rows (same OSM id and geometry) - i.e. true exact duplicates.

Run it directly::

    ./venv/bin/python dedupe_roads.py
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

SRC = Path(__file__).with_name("roads_within_5_miles.json")
BACKUP = Path(__file__).with_name("roads_within_5_miles.raw.json")


def name_key(road: dict) -> str:
    """Normalised road name used to decide whether two roads are the same."""
    return (road.get("name") or "").strip().casefold()


def name_speed_key(road: dict) -> tuple:
    return (name_key(road), (road.get("maxspeed") or "").strip().casefold())


def exact_key(road: dict) -> str:
    """Everything that identifies the exact OSM way (id + geometry)."""
    return json.dumps(
        [road.get("osm_id"), road.get("geometry_type"), road.get("coordinates")],
        sort_keys=True,
    )


def main() -> None:
    data = json.loads(SRC.read_text())
    roads = data["roads"]
    total = len(roads)

    # --- Analyse the three definitions ------------------------------------
    by_name = Counter(name_key(r) for r in roads)
    by_name_speed = Counter(name_speed_key(r) for r in roads)
    by_exact = Counter(exact_key(r) for r in roads)

    unique_name = len(by_name)
    unique_name_speed = len(by_name_speed)
    unique_exact = len(by_exact)

    print(f"File: {SRC}")
    print(f"Total road entries          : {total}")
    print()
    print("Duplicates by definition:")
    print(f"  same name                 : {total - unique_name} duplicated entries "
          f"-> {unique_name} unique roads")
    print(f"  same name + maxspeed      : {total - unique_name_speed} duplicated entries "
          f"-> {unique_name_speed} unique roads")
    print(f"  exact identical rows      : {total - unique_exact} duplicated entries "
          f"-> {unique_exact} unique roads")

    repeated = [(n, c) for n, c in by_name.items() if c > 1]
    repeated.sort(key=lambda x: x[1], reverse=True)
    print()
    print(f"Road names appearing more than once: {len(repeated)}")
    print("Most repeated names:")
    for name, count in repeated[:10]:
        print(f"  {count:4d}x  {name}")

    # --- Remove duplicates (keep the first entry for each name) -----------
    seen: set = set()
    kept: list[dict] = []
    for road in roads:
        key = name_key(road)
        if key in seen:
            continue
        seen.add(key)
        kept.append(road)

    # --- Save the de-duplicated result ------------------------------------
    if not BACKUP.exists():
        shutil.copyfile(SRC, BACKUP)
        print(f"\nBacked up original to: {BACKUP}")

    data["metadata"]["road_count"] = len(kept)
    data["metadata"]["duplicates_removed"] = total - len(kept)
    data["metadata"]["dedupe_definition"] = "same road name (case-insensitive)"
    data["roads"] = kept
    SRC.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    print(f"Wrote {len(kept)} unique roads to: {SRC}")
    print(f"\nUNIQUE ROADS: {len(kept)}")


if __name__ == "__main__":
    main()
