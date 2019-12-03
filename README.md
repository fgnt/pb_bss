# Blind Source Separation (BSS) algorithms

[![Build Status](https://dev.azure.com/fgnt/fgnt/_apis/build/status/fgnt.pb_bss?branchName=master)](https://dev.azure.com/fgnt/fgnt/_build/latest?definitionId=1&branchName=master)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/fgnt/pb_bss/master/LICENSE)

This repository covers EM algorithms to separate speech sources in multi-channel recordings.

In particular, the repository contains methods to integrate Deep Clustering (a neural network-based source separation algorithm) with a probabilistic spatial mixture model as proposed in the Interspeech paper "Tight integration of spatial and spectral features for BSS with Deep Clustering embeddings" presented at Interspeech 2017 in Stockholm.

```
@InProceedings{Drude2017DeepClusteringIntegration,
  Title                    = {Tight integration of spatial and spectral features for {BSS} with Deep Clustering embeddings},
  Author                   = {Drude, Lukas and and Haeb-Umbach, Reinhold},
  Booktitle                = {INTERSPEECH 2017, Stockholm, Sweden},
  Year                     = {2017},
  Month                    = {Aug}
}
```

# Installation
Install it directly from source
```bash
git clone https://github.com/fgnt/pb_bss.git
cd pb_bss
pip install --editable .
```
We expect that `numpy`, `scipy` and `cython` are installed (e.g. `conda install numpy scipy cython` or `pip install numpy scipy cython`).

In the `setup.py` we follow the concept of minimal dependencies.
So depending on which part of this core you are interested,
you need to install the following extra dependencies:

```bash
pip install -e .[metrics]  # Metric dependencies
pip install -e .[examples]  # Notebook (ipynb) example dependency
pip install -e .[tests]  # Test dependency
pip install -e .[mm]  # Bingham mixture model symbolic solution
# Or, to install all dependencies, use
pip install -e .[all]
```
