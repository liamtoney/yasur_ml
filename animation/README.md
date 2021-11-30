# Creating animations of Yasur data with cumulative catalog plots

The Python tool [_sonify_](https://github.com/liamtoney/sonify) can create animated
spectrograms with audio derived from sped-up waveforms. The code is readily hackable;
here we show how to modify the code to also generate a dynamic plot of cumulative
labeled events.

## Setup

First, ensure you're in this directory (`animation/`) and clone _sonify_ via
```shell
git clone https://github.com/liamtoney/sonify.git
```

Then, apply [`yasur.patch`](`yasur.patch`) to
[revision `f467ae1`](https://github.com/liamtoney/sonify/tree/f467ae1b3d2912fdfa2fdf395e050f0df7fc269c)
of _sonify_ via
```shell
cd sonify
git checkout f467ae1
git apply ../yasur.patch
```

Finally, install _sonify_ into the `yasur_ml` conda environment via
```shell
conda activate yasur_ml
pip install --editable .
```
(The `yasur_ml` environment already contains all of _sonify_'s dependencies.)

## Creating animations

Here's an example call:
```shell
sonify 3E YIF3 CDF 2016-07-31T19:00 2016-07-31T21:00 --freqmin 0.2 --freqmax 4 --speed_up_factor 400 --fps 60 --spec_win_dur 20 --db_lim 90 115
```
