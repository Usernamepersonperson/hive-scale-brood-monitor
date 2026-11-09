# Data sources and licences

| Data | Source | Licence | Used for |
|---|---|---|---|
| Hourly weather (temperature, precipitation, sunshine, radiation, wind) for Zurich / Fluntern, Bern / Zollikofen, Engelberg, 2023-2024 | MeteoSwiss, SwissMetNet open data, https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn | Open government data, attribution "Source: MeteoSwiss" | drives the simulator |
| Hourly hive-scale weight of 54 colonies (Denmark, Portugal), 2019-2020, plus beekeeper log (swarm, queen loss, feeding, harvest) | Zenodo, doi 10.5281/zenodo.5148318, "Research project on field data collection for honey bee colony model evaluation - datasets" | CC BY 4.0 | real swarm detection, real forecast check |
| 10-minute brood-nest temperature, humidity, weight of two Apis mellifera hives (Ceara, Brazil), Dec 2024 - Mar 2025 | Zenodo, doi 10.5281/zenodo.20399470 | CC BY 4.0 | real brood-temperature statistics |
| Annotated hive audio (queen present / absent) | Zenodo, doi 10.5281/zenodo.1321278, "To bee or not to bee" | CC BY 4.0 | acoustic experiment (features only are committed) |

Only small derived tables are committed (`hivemon/data/`): the weight table is a column subset of the
published one with bad timestamps (year 1970) dropped; the audio file holds band-power features per
10 s window, not audio. `python scripts/build_fixtures.py` rebuilds them from the sources.
Raw downloads (audio) belong in `cache/` (gitignored).

Not used: a large German citizen-science hive data set (Zenodo 10.5281/zenodo.10407693) has real
swarm labels, but its Zenodo record states no licence, so redistribution of derived results could
not be verified.

Beekeeper numbers: Agroscope, "Bienenhaltung in der Schweiz": about 17,500 beekeepers keep about
165,000 colonies, https://www.agroscope.admin.ch/de/bienenhaltung-in-der-schweiz
