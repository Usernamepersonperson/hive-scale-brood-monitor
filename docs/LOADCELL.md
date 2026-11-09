# Load-cell drift and temperature compensation

Model of a bar-type cell: `W_raw = W (1 + a dT) + z dT + creep + noise`, `dT = T_cell - 20 C`.
Simulated values: span coefficient a about 0.03 %/K, zero coefficient z about 8 g/K (author's
assumption of a cheap cell, not a datasheet value). With the cell in the sun, dT swings 10-20 K a
day, which is 0.2-0.4 kg of spurious weight, the size of a small nectar gain.

Compensation (`hivemon/sensor.py`): the node reports cell temperature next to weight. In 28-day
blocks, the slope of hourly weight change against hourly cell-temperature change is fitted on quiet
nights (23:00-04:00, shock below 0.4 g) with a Theil-Sen estimator after removing the per-hour
night profile, because evaporation and cooling both decay through the night and would otherwise
bias the slope. `W_comp = W_raw - k (T_cell - 20)`. Slow creep is not removed; it shows as a
long-term trend, so re-zero the scale when the hive is empty in winter.

Measured on the simulated cohort: see README (modest gain; the estimate of k is noisy).
Better: measure k once on the bench in a warm/cold cycle with a fixed weight and store it.
