# Simulator (all of this is SIMULATED data)

`hivemon/sim.py` simulates one hive-year at 15-minute resolution on real MeteoSwiss weather.
Parameters are plausible beekeeping magnitudes chosen by the author, not fitted to an apiary.

- Foraging suitability per step: logistic in temperature (centre 12 C), scaled by global radiation,
  cut to 10 % in rain, reduced above 5 m/s wind.
- Nectar flow potential per day: five Gaussian flows (fruit bloom, rape, meadow, linden, honeydew),
  shifted 3 days per 100 m above 550 m, random amplitude per hive.
- Nectar intake = flow x colony strength x suitability. Ripening: unripe pool loses 56 % of its
  mass as water (time constant 14 h at night, 48 h by day).
- Consumption: 15 g/day base + brood term (up to 0.22 kg/day) + cold term.
- Brood nest: 34.5 C regulated with small noise when brood is present; fluctuating cluster
  temperature when it is not. Sensor stability degrades when brood area is small.
- Scenarios: swarm (1-3 kg lost over 15-30 min on a warm dry midday, Apr-Jun), robbing (2-6 dearth
  days, 0.06-0.15 kg per 15 min), queen loss (brood gone 21 days later), theft (hive lifted, scale
  reads empty), tipping, plus routine inspections, super added, honey harvest, autumn feeding.
  Inspections raise a shock flag 90 % of the time and a service-mode flag 85 % of the time.
- Device view: load-cell drift (see LOADCELL.md), 10 g payload resolution is applied only in the
  payload encoder.

Labels are returned with every hive (`events`) and truth columns are prefixed `true_`. Analytics
never read the `true_` columns.
