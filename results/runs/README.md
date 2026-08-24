# Reproduction runs

`python scripts/reproduce_headline.py --gpu 0` writes a timestamped directory
here. It contains `summary.json`, the observed `environment.json`, all raw arm
artifacts under `raw/`, and captured process output under `logs/`.

Run directories are ignored by Git. A valid raw arm is reused on resume only
after its model, checkpoint revision, shape, dtype, arm, protocol, clock
telemetry, and sample count pass validation.
