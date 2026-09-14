# OSS elevation vs USGS / Cesium

Automations do **not** fetch.

| Source | License / lock | How Home may use it |
|---|---|---|
| Open-Elevation | OSS, self-host SRTM | Operator may run it on the box later. Not Docker-from-chat. |
| Nextzen / AWS Open Data terrain | keyless tiles | Offline GeoTIFF → numpy later. |
| USGS 3DEP / EPQS | portal | `dem_tile.py` exists; Gaia path sets `usgs_fetch=false`. |
| Cesium ion | keyed | Off. |

Live mesh tonight: `ports.terrain_spine.flagstaff_mesh` (~2100 m prior + ridges), then SVD reconstruct.
