# required libraries
import yaml
import pickle
import numpy as np
import pandas as pd

from lifelines import LogLogisticAFTFitter
from KaplanMeierEstimator import KaplanMeierEstimator

def load_config(config_path):
    '''Method to load config-file.'''
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_object_by_pickle(path, saved_obj):
    '''Method to save the object into file (serialization).'''
    s = pickle.dumps(saved_obj)
    fd = open(path, 'wb')
    fd.write(s)
    fd.close()

def example_how_to_train_survival_models(config):
    '''Example how to train survival models (You have to train YOUR models and REMOVED this function).'''
    # We describe a simulated dataset of patients that have some random covariates (=features)
    dataset = pd.DataFrame([
        {'id': 0, 'sex': 'M', 'age': 48, 'has_cancer': 0, 'visit_num': 12, 'has_D00_D48': 0, 'diagnosis_prop': 0.72},
        {'id': 1, 'sex': 'M', 'age': 59, 'has_cancer': 1, 'visit_num': 31, 'has_D00_D48': 0, 'diagnosis_prop': 0.46},
        {'id': 2, 'sex': 'M', 'age': 64, 'has_cancer': 0, 'visit_num': 22, 'has_D00_D48': 1, 'diagnosis_prop': 0.53},
        {'id': 3, 'sex': 'M', 'age': 67, 'has_cancer': 1, 'visit_num': 25, 'has_D00_D48': 1, 'diagnosis_prop': 0.58},
        {'id': 4, 'sex': 'M', 'age': 72, 'has_cancer': 0, 'visit_num': 18, 'has_D00_D48': 0, 'diagnosis_prop': 0.63},
        {'id': 5, 'sex': 'F', 'age': 52, 'has_cancer': 0, 'visit_num': 27, 'has_D00_D48': 0, 'diagnosis_prop': 0.68},
        {'id': 6, 'sex': 'F', 'age': 61, 'has_cancer': 0, 'visit_num': 32, 'has_D00_D48': 1, 'diagnosis_prop': 0.62},
        {'id': 7, 'sex': 'F', 'age': 66, 'has_cancer': 1, 'visit_num': 38, 'has_D00_D48': 0, 'diagnosis_prop': 0.44},
        {'id': 8, 'sex': 'F', 'age': 69, 'has_cancer': 1, 'visit_num': 35, 'has_D00_D48': 1, 'diagnosis_prop': 0.38},
        {'id': 9, 'sex': 'F', 'age': 75, 'has_cancer': 0, 'visit_num': 33, 'has_D00_D48': 1, 'diagnosis_prop': 0.63},
    ]).set_index('id')

    # Build the Kaplan-Meier estimator for MALES
    mask = dataset['sex'] == 'M'
    df = dataset[mask]
    T = df['age']
    C = 1 - df['has_cancer']                        # 0 - Failure, 1 - Right-Censored Observation
    km_males = KaplanMeierEstimator(T=T, C=C)

    # Build the Kaplan-Meier estimator for FEMALES
    mask = dataset['sex'] == 'F'
    df = dataset[mask]
    T = df['age']
    C = 1 - df['has_cancer']                        # 0 - Failure, 1 - Right-Censored Observation
    km_females = KaplanMeierEstimator(T=T, C=C)

    # Build the Kaplan-Meier estimator for MALES & FEMALES
    T = dataset['age']
    C = 1 - dataset['has_cancer']                   # 0 - Failure, 1 - Right-Censored Observation
    km_both = KaplanMeierEstimator(T=T, C=C)

    # Train the AFT model
    train = dataset.copy()
    train['has_cancer'] = 1 - train['has_cancer']   # 0 - Failure, 1 - Right-Censored Observation
    train['sex'] = train['sex'].apply(lambda sex: 1 if sex == 'M' else 0)

    aft = LogLogisticAFTFitter(
        alpha=0.05,
        fit_intercept=True
    ).fit(train, duration_col='age', event_col='has_cancer')

    aft_obj = {
        'model': aft,
        'covariates': ['sex', 'visit_num', 'has_D00_D48', 'diagnosis_prop'],
    }

    # Save these models
    save_object_by_pickle(config['path_kaplan_meier_males'], km_males)
    save_object_by_pickle(config['path_kaplan_meier_females'], km_females)
    save_object_by_pickle(config['path_kaplan_meier_both'], km_both)
    save_object_by_pickle(config['path_aft'], aft_obj)

# entry point
if __name__ == '__main__':
    # Load config-file
    config_path = './CONFIG_CanSave.yaml'
    config = load_config(config_path)

    # Start example to train survival models
    example_how_to_train_survival_models(config)
