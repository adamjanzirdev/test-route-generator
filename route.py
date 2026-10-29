"""Barebones route generator: OSRM driving route -> folium map -> HTML.

Uses the public OSRM demo server (built from OpenStreetMap road data).
To use your own local OSRM engine later, point cfg.host at it,
e.g. cfg.host = "http://localhost:5000".
"""
import osrm
import folium

# Start / end points as (latitude, longitude) from the user's request.
START = (51.249818284224936, -0.6681086431928808)  # lat, lon
END = (51.23437754112514, -0.5685526949288479)     # lat, lon

# OSRM works in (longitude, latitude).
origin = (START[1], START[0])
dest = (END[1], END[0])

# OSRM engine config (driving profile).
cfg = osrm.RequestConfig()
cfg.host = "https://routing.openstreetmap.de/routed-car"
cfg.version = "v1"
cfg.profile = "driving"

# Compute the driving route.
response = osrm.simple_route(origin, dest, geometry="geojson",
                             overview="full", url_config=cfg)
route = response["routes"][0]
line = route["geometry"]  # GeoJSON LineString

print(f"Route found: {route['distance']:.0f} m, {route['duration'] / 60:.1f} min")

# Folium map centered between start and end.
m = folium.Map(location=[(START[0] + END[0]) / 2, (START[1] + END[1]) / 2],
               zoom_start=12)

# Route line (coordinates come back as lon/lat).
route_coords = [(lat, lon) for lon, lat in line["coordinates"]]
folium.PolyLine(route_coords, color="blue", weight=5).add_to(m)

# Markers: "follow" because OSRM snap-points to the nearest road.
folium.Marker([START[0], START[1]], popup="Start",
              icon=folium.Icon(color="green")).add_to(m)
folium.Marker([END[0], END[1]], popup="End",
              icon=folium.Icon(color="red")).add_to(m)

m.save("route_map.html")
print("Saved route_map.html - open it in a browser.")
