# Blind Source Separation (BSS) algorithms

[![Build Status](https://dev.azure.com/fgnt/fgnt/_apis/build/status/fgnt.pb_bss?branchName=master)](https://dev.azure.com/fgnt/fgnt/_build/latest?definitionId=1&branchName=master)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/fgnt/pb_bss/master/LICENSE)

This repository covers EM algorithms to separate speech sources in multi-channel recordings.

In particular, the repository contains methods to integrate Deep Clustering (a neural network-based source separation algorithm) with a probabilistic spatial mixture model as proposed in the Interspeech paper "Tight integration of spatial and spectral features for BSS with Deep Clustering embeddings" presented at Interspeech 2017 in Stockholm.

```
@InProceedings{Drude2017DeepClusteringIntegration,
  Title                    = {{Tight integration of spatial and spectral features for BSS with Deep Clustering embeddings}},
  Author                   = {Drude, Lukas and and Haeb-Umbach, Reinhold},
  Booktitle                = {INTERSPEECH 2017, Stockholm, Sweden},
  Year                     = {2017},
  Month                    = {Aug}
}
```
