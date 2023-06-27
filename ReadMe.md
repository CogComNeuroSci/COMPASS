[![Uni](https://img.shields.io/badge/University-Ghent%20University-brightgreen)](https://img.shields.io/badge/University-Ghent%20University-brightgreen)
[![Date](https://img.shields.io/badge/Last%20update-2023-yellow)](https://img.shields.io/badge/Last%20update-2023-yellow)

# COmputational Power Analysis using Simulations "COMPASS" toolbox on DDM

Expanding COMPASS toolbox to DDM models.

More details about COMPASS can be found in the manuscript: *https://10.31234/osf.io/dexyk*


## core files
  InputFile_IC.csv: parameters for "angle" DDM
  PowerAnalysis.py: call power_estimation_Incorrelation() to analysis power
  Functions_DDM.py: call Incorrelation_repetition() to simulate, fit, and compute power
  trainLAN.py: train LAN on "angle" DDM with LANfactory and ssms