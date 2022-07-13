# yasur_ml

This repository contains the code accompanying the paper "Waveform features
strongly control subcrater classification performance for a large, labeled volcano infrasound
dataset" by [Liam Toney](mailto:ldtoney@alaska.edu), David Fee, Alex Witsil, and Robin S. Matoza.

## Installing

A conda environment specification file, [`environment.yml`](environment.yml), is
provided. You can create a conda environment from this file by executing
```shell
conda env create
```
from the repository root.

This code requires the [UAF Geophysics Tools](https://github.com/uafgeotools) package
[*rtm*](https://github.com/uafgeotools/rtm) for generation of the labeled
catalog. This package and its dependencies are installed when the above command
is executed.

You must define two environment variables to use the code:
- `YASUR_WORKING_DIR` — the path to this repository
- `YASUR_FIGURE_DIR` — the directory where figure files should be saved

## Workflow overview

1. [`download_3E.py`](data/download_3E.py) — download the data
2. [`build_catalog.py`](label/build_catalog.py) — run *rtm* to create a CSV catalog
3. [`label_catalog.py`](label/label_catalog.py) — associate entries in catalog to a subcrater
4. [`extract_features.py`](features/extract_features.py) — extract features from waveforms and store in Feather file
5. Apply tools in [`svm/`](svm/)

## Citation

If you use the tools contained in this repo, please cite our paper:

> Toney, L., Fee, D., Witsil, A., & Matoza, R. S. (in press). Waveform features
> strongly control subcrater classification performance for a large, labeled
> volcano infrasound dataset. *The Seismic Record*.

## Acknowledgements

This work was supported by the Nuclear Arms Control Technology (NACT) program at the
Defense Threat Reduction Agency (DTRA). Cleared for release.
