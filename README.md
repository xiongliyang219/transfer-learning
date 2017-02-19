# Transfer Learning on the Urban Tribes dataset

## Directory Structure

- `data` stores data sets (not tracked in git).
- `examples` contains several notebooks recording our early explorations.
- `results` stores results (tracked on [Google Drive](https://drive.google.com/open?id=0B1b89t9Bpw8TWXlod0VMNnlocGs)).
- `scripts` stores scripts to be run to get results.
- `tools` and `setup.py` constitute an installable Python package where organized codes live.

## Reproduce Instructions

To reproduce our results, first make sure the following dependencies are satisfied:

- python >= 3.5
- h5py
- keras
- matplotlib
- numpy
- tensorflow

Then install the `tools` package through:

```bash
pip install -e .
```

Then `cd` into the `scripts` directory and do:

```bash
# Train different models and store the resulting histories.
# This may take a long time.
python run.py

# Make plots.
python plot_history.py
python plot_filter.py
```

Parameters controlling theses scripts are stored in [`params.json`](scripts/params.json).
