# SN_environment

## Table of contents
* [General info](#general-info)
* [Content](#content)
* [References](#references)

## General info

Type Ia Supernovae are widely used to measure distances in the Universe. Despite the recent progress achieved in SN Ia standardization, the Hubble diagram still shows some remaining intrinsic dispersion. The remaining scatter in supernova luminosity could be due to the environmental effects.

In these Jupyter Notebooks we reproduce the Hubble diagram fit with Pantheon supernovae ([Scolnic et al., 2018](https://ui.adsabs.harvard.edu/abs/2018ApJ...859..101S/abstract)). We also study how the host morhology term and the galactocentric distance affect the supernova light-curve parameters and the Hubble diagram fit.


## Content

### Data

Contains data files
* `FITOPT000.FITRES` – file with those Pantheon supernovae that have photometry from two different sources, i.e. two sets of SALT2 LCs parameters are given.
* `Ancillary_G10.FITRES` – Pantheon data for G10 intrinsic scatter model.
* `Ancillary_C11.FITRES` – Pantheon data for C11 intrinsic scatter model.
* `sys_full_long_G10.txt` – systematic covariance matrix for G10 intrinsic scatter model.
* `sys_full_long_C11.txt` – systematic covariance matrix for C11 intrinsic scatter model.
* `lcparam_full_long.txt` – file with the final distances (add ~19.35 to them to get μ values).


### Code

* `SNIa_Pantheon_fit.ipynb` – Hubble diagram fit of Pantheon supernovae.
* `comparison_mu.ipynb` – comparison between distance modulus *'MU'* from Pantheon data and distance modulus calculated from from *'mB', 'x1', 'c',* and other parameters taken from Pantheon.
* `pantheon_morphology.ipynb` – shows how host morphology and galactocentric distance affect the supernova light-curve parameters.

### Plots
Output plots from Jupyter Notebooks.

## References
* Pruzhinskaya, M. V., Novinskaya, A. K., Pauna, N., Rosnet, P. "The dependence of Type Ia Supernovae SALT2 light-curve parameters on host galaxy morphology", [MNRAS, Vol. 499, Issue 4, pp.5121-5135, 2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.5121P/abstract)

* Pruzhinskaya, M., Novinskaya, A., Rosnet, P., Pauna, N. "Influence of host galaxy morphology on the properties of Type Ia supernovae from JLA and Pantheon compilations", [Proceedings of Science, Multifrequency Behaviour of High Energy Cosmic Sources Workshop- XIII (MULTIF2019), Palermo, Italy, 362, 2020](https://ui.adsabs.harvard.edu/abs/2020mbhe.confE..15P/abstract)
