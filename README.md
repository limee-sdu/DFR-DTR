# DFR-DTR
This repository contains the official PyTorch implementation of the following paper:

## Requirement
We recommended the following dependencies.

- Python 3.9.0
- PyTorch (1.10.0)
- NumPy (1.23.5)
- Matplotlib (3.6.2)
## Download data
### MIMIC-III dataset
Although de-identified, the datasets described herein contain detailed information regarding the clinical care of patients, and as such it must be treated with appropriate care and respect. Researchers seeking to use the database must formally request access. Details are shown on
 [the MIMIC website](https://mimic.mit.edu/).
### eICU dataset
The eICU Collaborative Research Database contains detailed information regarding the clinical care of ICU patients, so it must be treated with appropriate care and respect. Researchers seeking to use the database must formally request access. Details are shown on [the eICU website](https://eicu-crd.mit.edu/).

## Preprocessing
Here we give the source code of the preprocessing for eICU dataset.

## Train model
You can use the following command to run the `main.py` on eICU dataset

`python main.py --epoch-num 7 --actor-lr 0.1 --cluster-num 6`
