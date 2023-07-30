# AHAAB
**A**ntigen-**H**LA-**A**ffinity Predictions from **A**tom-**B**ased Features

## Publication

## Authors
Nathaniel Bloodworth, MD, PhD
Vanderbilt University Medical Center
Department of Medicine
Division of Clinical Pharmacology
Nashville, TN

Developed in collaboration with the laboratory and resources of Jens Meiler, PhD

[Github repository](https://www.github.com/nbloodworth/ahaab)

# Introduction
The goal of **AHAAB** is to adderess deficiencies in current tools that use primarily sequence-based or amino-acid specific features for predicing antigen-HLA binding affinity. By featurizing models of peptides bound to HLA variants, **AHAAB** can accurately predict binding affinity for a wider variety of antigens containing post-translational modifications or other less common amino acid variants. 

# Installation

# Modules

ahaab/src/
├──ahaab.py
├──features/
|   ├──get_features.py
|   └──atom_features.py
├──train/
|   └──train_ahaab.py
├──predict/
|   └──classifiers.py
├──tools/
|   ├──validate_input.py
|   ├──retrieve_data.py
|   ├──multitask.py
|   └──formats.py
└──data/
    ├──features/
    └──weights/

## Dependencies

