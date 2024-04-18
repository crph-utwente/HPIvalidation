## Introduction
This is the accompanying analysis code for the article *'The Hypotension Prediction Index is equally effective in predicting intraoperative hypotension during non-cardiac surgery compared to a mean arterial pressure threshold: a prospective observational study.'*

When using (part of) this code, please cite the acompanying article: "Mulder MP, Harmannij-Markusse M, Fresiello L, Donker DW, Potters JW. The Hypotension Prediction Index is equally effective in predicting intraoperative hypotension during non-cardiac surgery compared to a mean arterial pressure threshold: a prospective observational study. Anesthesiology. 2024 Apr 1. doi: 10.1097/ALN.0000000000004990."

Abbreviations:
- AUT:      area under the treshold
- dMAP:     delta mean arterial pressure
- FP:       false positive
- HPI:      Hypotension Prediction Index
- IOH:      intraoperative hypotension
- lepMAP:   linearly extrapolated mean arterial pressure
- MAP:      mean arterial pressure 
- TP:       true positive
- TWA:      time weigthed average

## Usage
This repository contains the following files
- `main_analysis.py` : Script to analyse multiple patients from the *input_data* directory. It is important to specify the start and end of the surgery of each individual patient. This is denoted in the *timepoints* dictionary in this file. Adapt to your own data format. There are multiple formats in which the source data can exist. This script will check two different versions, but it might fail if the data format contains specific local datetime data.

- `main_analysis_notebook.ipynb` : Jupyter Notebook where each analysis step is documented for one individual patient. Anonimized example data is provided in the file `input_data/example_patient.xls`


The specific requirements to run this script are listed in the file `requirements.txt`. This can be installed to your local Python environment by running:
```
pip install -r requirements.txt
```

---
This work © 2024 by CRPH group of the University of Twente is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International.
