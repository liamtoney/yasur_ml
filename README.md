# yasur_ml

This repository contains the code accompanying the forthcoming manuscript "Influence of
Waveform Feature Extraction on Machine Learning Classifier Performance for a Large,
Labeled Volcano Infrasound Dataset" [draft title] by
[Liam Toney](mailto:ldtoney@alaska.edu), [David Fee](mailto:dfee1@alaska.edu),
[Alex Witsil](mailto:ajwitsil@alaska.edu), and [Robin Matoza](mailto:rmatoza@ucsb.edu).

## Installing

A conda environment specification file, [`environment.yml`](environment.yml), is
provided. You can create a conda environment from this file by executing
```shell
conda env create
```
from the repository root.

This code requires the [UAF Geophysics Tools](https://github.com/uafgeotools) package
[*rtm*](https://github.com/uafgeotools/rtm) for generation of the labeled
catalog. You can find installation instructions for *rtm*
[here](https://uaf-rtm.readthedocs.io/en/master/README.html#installation).
