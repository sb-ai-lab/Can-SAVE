# Can-SAVE: *Deploying Low-Cost and Population-Scale Cancer Screening via Survival Analysis Variables and EHR*

[![arXiv](https://img.shields.io/badge/arXiv-2309.15039-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2309.15039)
[![KDD 2026](https://img.shields.io/badge/KDD%202026-Accepted-2ea44f?logo=acm)](https://kdd2026.kdd.org/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?logo=python)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

The source code to implement the feature engineering step of the Can-SAVE method.

## Installation
```bash
git clone https://github.com/sb-ai-lab/CanSave.git
cd CanSave
pip install -r requirements.txt
```

## requirements.txt
```bash
pandas==1.5.3
numpy==1.23.2
lifelines==0.27.4
scikit-learn==1.1.3
scipy==1.10.0
PyYAML==6.0
openpyxl==3.0.10
```

## Repository Structure
- Can-SAVE/: Core implementation
- EHR/: Simulated sample of EHR data
- survival_models/: Output directory for fitted models (Kaplan-Meier estimators and AFT model)

```bash
Can-SAVE/
├── EHR/
│   └── id_26.csv
├── survival_models/
│   ├── kaplan_meier_both.pkl
│   ├── kaplan_meier_males.pkl
│   ├── kaplan_meier_females.pkl
│   └── aft.pkl
├── CanSave.py
├── Example_How_To_Train_Survival_Models.py
├── KaplanMeierEstimator.py
├── CONFIG_CanSave.yaml
├── icd10_groups.xlsx
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick Start

### 1) How to Train Survival Models
```bash
$ python Example_How_To_Train_Survival_Models.py
```

### 2) How to Do Feature Engineering for Can-SAVE
#### Terminal
```bash
$ python CanSave.py
```

#### Python
```python
# required libraries
import numpy as np
import pandas as pd

from CanSave import CanSave

# entry point
if __name__ == '__main__':
    # Make new object for feature engineering
    config_path = './CONFIG_CanSave.yaml'
    cs = CanSave(CONFIG_PATH=config_path)
    print(help(cs))

    # Load the patient's EHR
    path_ehr = './EHR/id_26.csv'
    ehr = pd.read_csv(path_ehr, sep=';').set_index('patient_id')
    sex = ehr['sex'].iloc[0]
    birth_date = ehr['birth_date'].iloc[0]

    # Make feature engineering for the risk prediction
    features = cs.feature_engineering(
        sex         = sex,              # sex of the patient
        birth_date  = birth_date,       # birth date of the patient
        ehr         = ehr,              # Electronic Health Records of the patient
        date_pred   = '2022-01-01',     # date of the risk estimation
        deep_weeks  = 108               # deep of the EHR's history (in weeks)
    )

```
