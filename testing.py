"""List the *drivable* roads within a radius of a point, from a local OSM PBF file.

The heavy lifting is done with ``pyrosm`` (reads the ``.osm.pbf``), ``shapely`` /
``pyproj`` (distance / connectivity maths) and ``geopandas`` (tabular + spatial
data).

Roads are kept only when they are:

* drivable (their ``highway`` value is in :data:`DRIVABLE_HIGHWAYS`),
* within ``RADIUS_MILES`` of the centre,
* not a dead end - both ends join another drivable road,
* tagged with a non-empty ``name``, and
* tagged with a non-empty ``maxspeed``.

For each surviving road *all* of its tags are kept, so properties such as
``lanes``, ``surface`` and ``oneway`` are included when they exist.  The result
is written to a human-readable, vertically formatted JSON file; the terminal
prints just the road names and their speed limits.

Run it directly::

    ./venv/bin/python testing.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyrosm
from pyproj import CRS, Transformer
from pyrosm import OSM
from shapely.geometry import Point, mapping

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Local OpenStreetMap extract to read from.
PBF_PATH = Path(__file__).with_name("united-kingdom-261027.osm.pbf")

# Centre of the search as (latitude, longitude).
CENTER_LAT = 51.27387596167663
CENTER_LON = -0.771035741259303

# Search radius in statute miles.
RADIUS_MILES = 5.0
# 1 international mile = 1609.344 m.
METERS_PER_MILE = 1609.344
RADIUS_M = RADIUS_MILES * METERS_PER_MILE

# The PBF is read from a slightly larger box than the search circle.  The extra
# ring gives every road near the edge a chance to "see" the roads it connects to
# just outside the circle, so boundary roads are not mistaken for dead ends.
READ_MARGIN_M = METERS_PER_MILE  # read 1 mile beyond the radius.

# Where to save the structured result.
OUTPUT_JSON = Path(__file__).with_name("roads_within_5_miles.json")

# ``highway`` values that a motor vehicle can drive on.  Anything else (footways,
# paths, cycleways, steps, ...) is not a "drivable road".
DRIVABLE_HIGHWAYS = [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "road",
]

# A metric CRS suited to the UK, used to measure distances/lengths in metres.
# (British National Grid.)
METRIC_CRS = CRS.from_epsg(27700)
WGS84 = CRS.from_epsg(4326)

# Commonly useful road tags promoted to their own columns.  Any tag *not* listed
# here is still preserved inside the JSON ``tags`` column, so nothing is lost.
ROAD_TAG_COLUMNS = [
    "name",
    "ref",
    "highway",
    "maxspeed",
    "maxspeed:type",
    "maxspeed:forward",
    "maxspeed:backward",
    "oneway",
    "lanes",
    "lanes:forward",
    "lanes:backward",
    "surface",
    "smoothness",
    "width",
    "access",
    "motor_vehicle",
    "motorcar",
    "bicycle",
    "foot",
    "service",
    "junction",
    "bridge",
    "tunnel",
    "layer",
    "lit",
    "sidewalk",
    "cycleway",
    "operator",
    "designation",
    "destination",
    "postal_code",
    "road_class",
    "noexit",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bounding_box(center_lat: float, center_lon: float, radius_m: float) -> list[float]:
    """Return ``[min_lon, min_lat, max_lon, max_lat]`` for a circle around a point.

    The circle's bounding rectangle is built in the metric CRS, then its corners
    are converted back to WGS84 so pyrosm can pre-filter on it.
    """
    to_metric = Transformer.from_crs(WGS84, METRIC_CRS, always_xy=True)
    to_wgs = Transformer.from_crs(METRIC_CRS, WGS84, always_xy=True)
    cx, cy = to_metric.transform(center_lon, center_lat)

    lons, lats = [], []
    for x, y in (
        (cx - radius_m, cy - radius_m),
        (cx - radius_m, cy + radius_m),
        (cx + radius_m, cy - radius_m),
        (cx + radius_m, cy + radius_m),
    ):
        lon, lat = to_wgs.transform(x, y)
        lons.append(lon)
        lats.append(lat)

    margin = 0.01  # ~1 km of slack in degrees; just widens the read box slightly.
    return [min(lons) - margin, min(lats) - margin, max(lons) + margin, max(lats) + margin]


def _clean(value):
    """Turn pandas/numpy scalars into plain JSON-serialisable Python values."""
    if value is None:
        return None
    try:
        if value != value:  # NaN check that also handles pandas NA.
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):  # numpy scalar types.
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _merge_tags(row) -> dict:
    """Collect every OSM tag for a road into a single dict.

    Tags promoted to their own columns and the JSON ``tags`` column (which holds
    everything else) are merged; explicit columns win on conflicts.
    """
    tags: dict = {}

    # The catch-all JSON column first, so column values can override it.
    raw = row.get("tags")
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                tags.update(parsed)
        except json.JSONDecodeError:
            pass
    elif isinstance(raw, dict):
        tags.update(raw)

    for key in ROAD_TAG_COLUMNS:
        if key in row:
            cleaned = _clean(row.get(key))
            if cleaned is not None:
                tags[key] = cleaned

    return tags


def _non_empty(value) -> bool:
    """True when ``value`` is a string with actual content (not None/NaN/blank)."""
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in ("nan", "none")


def _connected_both_ends(ways) -> "pd.Series":
    """Return a boolean Series marking ways that are *not* dead ends.

    A way is connected when **both** of its end points are also part of at least
    one *other* way in ``ways`` (i.e. the road joins another road at each end
    rather than stopping in a cul-de-sac).  Connectivity is inferred from the
    geometry: every vertex shared by two ways is a junction, and a free
    (unshared) end point means the road dead-ends there.
    """
    import numpy as np
    from shapely import get_coordinates

    ids = ways["id"].to_numpy()
    coords_per_way = []
    endpoint_keys: set = set()

    for geom in ways.geometry.to_numpy():
        c = get_coordinates(geom)
        coords_per_way.append(c)
        if len(c):
            endpoint_keys.add((round(float(c[0, 0]), 7), round(float(c[0, 1]), 7)))
            endpoint_keys.add((round(float(c[-1, 0]), 7), round(float(c[-1, 1]), 7)))

    # Which ways own each coordinate that is somebody's end point.
    owners: dict = {}
    for wid, c in zip(ids, coords_per_way):
        for x, y in c:
            key = (round(float(x), 7), round(float(y), 7))
            if key in endpoint_keys:
                owners.setdefault(key, set()).add(wid)

    flags = np.zeros(len(ways), dtype=bool)
    for i, (wid, c) in enumerate(zip(ids, coords_per_way)):
        if not len(c):
            continue
        e1 = (round(float(c[0, 0]), 7), round(float(c[0, 1]), 7))
        e2 = (round(float(c[-1, 0]), 7), round(float(c[-1, 1]), 7))
        joins_start = any(other != wid for other in owners.get(e1, ()))
        joins_end = any(other != wid for other in owners.get(e2, ()))
        flags[i] = joins_start and joins_end

    return pd.Series(flags, index=ways.index)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def find_roads_within_radius(
    pbf_path: Path | str = PBF_PATH,
    center_lat: float = CENTER_LAT,
    center_lon: float = CENTER_LON,
    radius_miles: float = RADIUS_MILES,
    output_path: Path | str | None = OUTPUT_JSON,
    verbose: bool = True,
    progress: bool = False,
) -> list[dict]:
    """Return the drivable, named, speed-limited roads within ``radius_miles``.

    Only roads that are drivable, lie inside the circle, join another road at
    both ends (i.e. are not dead ends), and have both a ``name`` and a
    ``maxspeed`` tag are returned.

    Parameters
    ----------
    pbf_path:
        Path to a local ``.osm.pbf`` extract.
    center_lat, center_lon:
        Centre of the search (WGS84 degrees).
    radius_miles:
        Search radius in statute miles.
    output_path:
        Optional path to write a pretty-printed JSON list of roads to.
    verbose:
        Print the road list and the total count to stdout.
    progress:
        Print status messages (file, counts, saved path).  Off by default so
        the terminal shows only the roads and the total.

    Returns
    -------
    list[dict]
        One dict per road, sorted by name.  Each dict holds the OSM id, a
        ``tags`` mapping with all known properties, and the road geometry as
        GeoJSON coordinates.
    """
    radius_m = radius_miles * METERS_PER_MILE

    if not progress:
        # Keep the terminal clean: only the roads and the total are printed.
        pyrosm.enable_logging("WARNING", progress=False)

    if progress:
        print(f"PBF file       : {pbf_path}")
        print(f"Centre         : {center_lat}, {center_lon}")
        print(f"Radius         : {radius_miles} miles ({radius_m:.0f} m)")

    # 1) Rectangular pre-filter (radius + margin), so pyrosm only keeps roads we
    #    might need and every edge road can still see its neighbours.
    bbox = _bounding_box(center_lat, center_lon, radius_m + READ_MARGIN_M)
    if progress:
        print(f"Bounding box   : {[round(v, 5) for v in bbox]}  (min_lon, min_lat, max_lon, max_lat)")
        print("Reading OSM data (this can take a while for a large extract)...")

    osm = OSM(str(pbf_path), bounding_box=bbox)

    # 2) Read only drivable highways.  Any tag not listed in ROAD_TAG_COLUMNS is
    #    still kept inside the JSON `tags` column (the default).
    network = osm.get_data_by_custom_criteria(
        custom_filter={"highway": DRIVABLE_HIGHWAYS},
        tags_as_columns=ROAD_TAG_COLUMNS,
        keep_nodes=False,
        keep_ways=True,
        keep_relations=False,
        keep_other_tags=True,
    )

    if network is None or len(network) == 0:
        if progress:
            print("No drivable roads found in the bounding box.")
        return []

    # Drop rows without usable geometry.
    network = network[network.geometry.notna() & ~network.geometry.is_empty].copy()

    if progress:
        print(f"Drivable ways  : {len(network)} (radius + margin box)")

    # 3) Dead-end test against the whole read network, so roads near the edge are
    #    judged against the roads they connect to just outside the circle.
    connected = _connected_both_ends(network)

    # 4) Precise circular filter: project to metres and measure the shortest
    #    distance from the centre to each road.
    to_metric = Transformer.from_crs(WGS84, METRIC_CRS, always_xy=True)
    cx, cy = to_metric.transform(center_lon, center_lat)
    center_metric = Point(cx, cy)
    network_metric = network.to_crs(METRIC_CRS)
    distance_m = network_metric.geometry.distance(center_metric)

    roads = network[distance_m <= radius_m].copy()
    distance_m = distance_m.loc[roads.index]

    if progress:
        print(f"In circle      : {len(roads)}")

    # 5) Keep only roads connected at both ends, with a real name and a real
    #    speed limit, and not explicitly tagged as a dead end (`noexit=yes`).
    keep_connected = connected.loc[roads.index]
    is_named = roads["name"].map(_non_empty) if "name" in roads else False
    has_speed = roads["maxspeed"].map(_non_empty) if "maxspeed" in roads else False
    noexit = (
        roads["noexit"].astype(str).str.strip().str.lower().eq("yes")
        if "noexit" in roads
        else False
    )
    roads = roads[keep_connected & is_named & has_speed & ~noexit].copy()
    distance_m = distance_m.loc[roads.index]
    network_metric = network_metric.loc[roads.index]  # keep geometry for lengths

    if progress:
        print(f"After filters  : {len(roads)}")

    # 6) Build a clean, fully-tagged record per road.
    records: list[dict] = []
    for idx, row in roads.iterrows():
        tags = _merge_tags(row)
        geom = row.geometry
        records.append(
            {
                "osm_id": _clean(row.get("id")),
                "name": tags.get("name"),
                "highway": tags.get("highway"),
                "maxspeed": tags.get("maxspeed"),
                "length_m": round(float(network_metric.geometry.loc[idx].length), 1),
                "distance_from_center_m": round(float(distance_m.loc[idx]), 1),
                "geometry_type": geom.geom_type,
                "coordinates": mapping(geom)["coordinates"],
                "tags": tags,
            }
        )

    # 7) Sort alphabetically by name, then nearest-first.
    records.sort(key=lambda r: ((r["name"] or "").lower(), r["distance_from_center_m"]))

    if output_path is not None:
        payload = {
            "metadata": {
                "center": {"lat": center_lat, "lon": center_lon},
                "radius_miles": radius_miles,
                "radius_meters": round(radius_m, 1),
                "pbf_file": str(pbf_path),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "road_count": len(records),
                "filters": {
                    "drivable_highway_values": DRIVABLE_HIGHWAYS,
                    "requires_name": True,
                    "requires_maxspeed": True,
                    "excludes_dead_ends": True,
                },
                "notes": (
                    "Each road is a drivable OSM way inside the radius that joins "
                    "another road at both ends and has a name and a maxspeed tag. "
                    "'tags' contains every tag found on the way; 'coordinates' is "
                    "GeoJSON [lon, lat]."
                ),
            },
            "roads": records,
        }
        output_path = Path(output_path)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        if progress:
            print(f"Saved          : {output_path}")

    if verbose:
        _print_summary(records)

    return records


def _print_summary(records: list[dict]) -> None:
    """Print every road, one per line, then the total count.

    Each road line is ``<name> : <maxspeed>``; the final line is the total.
    """
    for road in records:
        print(f"{road['name']} : {road['maxspeed']}")
    print(f"\nTotal roads: {len(records)}")


if __name__ == "__main__":
    find_roads_within_radius()

