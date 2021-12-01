# Creating animations of Yasur data with cumulative catalog plots

The Python tool [_sonify_](https://github.com/liamtoney/sonify) can create animated
spectrograms with audio derived from sped-up waveforms. The tool is readily hackable;
here we show how to modify the code to also generate a dynamic plot of cumulative
labeled events.

## Setup

First, ensure you're in this directory (`animation/`) and clone _sonify_ via
```text
git clone https://github.com/liamtoney/sonify.git
```

Then, apply [`yasur.patch`](yasur.patch) to
[revision `f467ae1`](https://github.com/liamtoney/sonify/tree/f467ae1b3d2912fdfa2fdf395e050f0df7fc269c)
of _sonify_ via
```text
cd sonify
git checkout f467ae1
git apply --unidiff-zero ../yasur.patch
```

Finally, install _sonify_ into the `yasur_ml` conda environment via
```text
conda activate yasur_ml
conda install -c conda-forge 'setuptools<58.4.0'  # To avoid a warning
pip install --editable .
```
(The `yasur_ml` environment already contains all of _sonify_'s dependencies.)

## Creating animations

The patched code looks for a Feather file containing features in `features/feather/`.
It reads this file and extracts the labeled events (`time` and `label` columns). You
must have already generated at least one of these files to run the code below! (If
there are multiple files present, the code will use the alphabetically first one.)

Here's an example call:
```text
sonify 3E YIF3 CDF 2016-07-31T19:00 2016-07-31T21:00 --freqmin 0.2 --freqmax 4 --speed_up_factor 400 --fps 30 --spec_win_dur 20 --db_lim 90 115
```
