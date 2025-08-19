# required libraries
import re
import yaml
import pickle

import numpy as np
import pandas as pd

from lifelines import LogLogisticAFTFitter
from KaplanMeierEstimator import KaplanMeierEstimator

import warnings
warnings.filterwarnings('ignore')

class CanSave:
    '''Object of the Can-Save method for feature engineering.'''
    def __load_object_by_pickle(self, path):
        '''Method to load a deserializated file.'''
        loaded_obj = pickle.load(open(path, 'rb'))
        return loaded_obj

    def __load_config(self, config_path):
        '''Method to load config-file.'''
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def __load_ICD10_groups(self):
        '''Method to load the table of ICD-10 groups.'''
        path_icd10_groups = self.config['path_icd10_groups']
        groups = pd.read_excel(path_icd10_groups).dropna(subset=['left_code', 'right_code'])
        groups = groups[groups['selected'] == 1]
        self.groups = groups

    def __load_survival_models(self):
        '''Method to load YOUR trained survival models.'''

        '''
            DISCLAIMER!
            We do not make trained models publicly available.  
            Therefore, we only provide an example of how survival models can be trained. 
            Example is located in the "Example_How_To_Train_Survival_Models.py".
        '''

        # Load trained survival models
        kaplan_meier_males = self.__load_object_by_pickle(self.config['path_kaplan_meier_males'])
        kaplan_meier_females = self.__load_object_by_pickle(self.config['path_kaplan_meier_females'])
        kaplan_meier_both = self.__load_object_by_pickle(self.config['path_kaplan_meier_both'])
        aft = self.__load_object_by_pickle(self.config['path_aft'])

        # Form dict of survival models
        self.survival_models = {
            'kaplan_meier_males': kaplan_meier_males,
            'kaplan_meier_females': kaplan_meier_females,
            'kaplan_meier_both': kaplan_meier_both,
            'aft': aft,
        }

    def __init__(self, CONFIG_PATH):
        '''Constructor.'''
        # Load config-file
        self.__load_config(CONFIG_PATH)
        # Load table of ICD-10 groups
        self.__load_ICD10_groups()
        # Load trained survival models
        self.__load_survival_models()

    def __check_russian_service_code(self, code):
        '''Method to check the medical service code used in the Russian Federation (e.g.: A04.16, B01.042).'''
        code = str(code).upper()
        mas = code.split('.')
        first = mas[0]
        second = mas[1] if len(mas) > 1 else '00'

        if 'A' in first:
            length = 2
        elif 'B' in first:
            length = 3
        else:
            raise ValueError(f'Invalid medical code: {code}')

        if len(first) < 3:
            first = first[0] + '0' + first[1]

        if len(second) == length:
            pass
        elif len(second) > length:
            second = second[-length:]
        else:
            while len(second) != length:
                second = '0' + second

        code = '{}.{}'.format(first, second)

        return code

    def __common_features(self, features, ehr, date_start, date_pred):
        '''Method to make common features.'''
        # MONTH OF THE PREDICTION
        month_pred = date_pred.month
        features['month_pred'] = month_pred

        # DAY OF WEEK
        features['day_of_week'] = date_pred.isoweekday()

        # VISIT_NUM
        visit_num = len(ehr)
        features['visit_num'] = visit_num

        # PROPORTION OF DIAGNOSIS AND SERVICES
        features['diagnosis_prop'] = ehr['is_diagnose'].mean()
        features['services_prop'] = 1.0 - ehr['is_diagnose'].mean()

        # WEEKS AFTER FIRST VISIT
        features['weeks_after_first_visit'] = (date_pred - ehr['date'].min()).days / 7.

        # WEEKS AFTER LAST VISIT
        features['weeks_after_last_visit'] = (date_pred - ehr['date'].max()).days / 7.

        # AVG. WEEKS BETWEEN VISITS
        features['weeks_betw_visits_avg'] = (ehr['date'].max() - ehr['date'].min()).days / (visit_num + 1.0e-14)

        # VISITS IN LAST N MONTHS
        for month in [1, 3, 6, 9, 12, 15, 18, 21, 24]:
            end_dt = date_pred
            start_dt = date_pred - pd.DateOffset(months=month)
            mask = (start_dt <= ehr['date']) & (ehr['date'] <= end_dt)

            new_col = f'visits_in_last_{month}_months'
            features[new_col] = mask.sum()

        # DIAGNOSE GROUPS
        mask = ehr['is_diagnose'] == 1
        df = ehr[mask]
        df['code_short'] = df['code'].apply(lambda code: str(code).split('.')[0])

        for left_mkb, right_mkb in zip(self.groups['left_code'], self.groups['right_code']):
            mask = (left_mkb <= df['code_short']) & (df['code_short'] <= right_mkb)
            new_col = 'diagnose_group_{}_{}'.format(left_mkb, right_mkb)
            features[new_col] = mask.sum()

        # UNIQUE DIAGNOSIS
        features['unique_diagnosis'] = len(set(df['code_short']))

        # UNIQUE CLASSES OF DIAGNOSIS
        df['code_class'] = df['code_short'].apply(lambda code: re.match('[A-Z]', str(code)).group(0))
        features['unique_classes_of_diagnosis'] = len(set(df['code_class']))

        # BINARY (1 - Diagnosis group has been in the EHR, 0 - has not been in the EHR)
        for left_mkb, right_mkb in zip(self.groups['left_code'], self.groups['right_code']):
            feature = f'diagnose_group_{left_mkb}_{right_mkb}'
            val = features[feature]

            new_col = f'has_{left_mkb}_{right_mkb}'
            features[new_col] = 1 if val > 0 else 0

        # WEEKS FROM LAST AND FIRST OCCURRENCE OF DIAGNOSIS GROUP
        for left_mkb, right_mkb in zip(self.groups['left_code'], self.groups['right_code']):
            mask = (left_mkb <= df['code']) & (df['code'] <= right_mkb)
            df_group = df[mask]

            new_col = f'weeks_from_first_{left_mkb}_{right_mkb}'
            if len(df_group) > 0:
                features[new_col] = (date_pred - df_group['date'].min()).days / 7.
            else:
                features[new_col] = (date_pred - date_start).days / 7.

            new_col = f'weeks_from_last_{left_mkb}_{right_mkb}'
            if len(df_group) > 0:
                features[new_col] = (date_pred - df_group['date'].max()).days / 7.
            else:
                features[new_col] = (date_pred - date_start).days / 7.

        return features

    def __regional_features(self, features, ehr, date_start, date_pred):
        '''
            Method to make features adopted to some regional specific characteristics.
            This method should be adopted for your country.
        '''
        # AVG. HUMIDITY AT MONTH FOR THE PREDICTION IN RUSSIA
        avg_humidity = {
            1: 88., 2: 84., 3: 74., 4: 65., 5: 64., 6: 68.,
            7: 72., 8: 69., 9: 77., 10: 81., 11: 88., 12: 91.
        }
        month_pred = features['month_pred']
        features['avg_humidity'] = avg_humidity[month_pred]

        # AVG. TEMPERATURE AT MONTH FOR THE PREDICTION IN RUSSIA
        avg_temperature = {
            1: -7.2, 2: -5.2, 3: -0.5, 4: 8.1, 5: 15.5, 6: 18.0,
            7: 20.3, 8: 18.9, 9: 13.3, 10: 5.7, 11: 1.3, 12: -3.4
        }
        features['avg_temperature'] = avg_temperature[month_pred]

        # SERVICE GROUPS (in according to ORDER 804n of the Ministry of Health of RUSSIA)
        mask = ehr['is_diagnose'] == 0
        df = ehr[mask]
        df['code_mod'] = df['code'].apply(self.__check_russian_service_code)

        # type of medical service
        codes = [f'A0{num}.' if num < 10 else f'A{num}.' for num in range(30+1)] + ['B01.','B02.','B03.','B04.','B05.']
        for val in codes:
            mask = df['code_mod'].str.contains(val)
            new_col = 'service_group_{}XX'.format(val)
            features[new_col] = mask.sum()

        # anatomical and functional area of medical service & list of medical specialties
        group1 = [f'.0{num}' if num < 10 else f'.{num}' for num in range(1, 30+1)]
        group2 = [f'.00{num}' if num < 10 else f'.0{num}' for num in range(1, 70+1)]
        codes = group1 + group2
        for val in codes:
            mask = df['code_mod'].str.contains(val)
            new_col = 'service_group_Axx{}'.format(val)
            features[new_col] = mask.sum()

        return features

    def __survival_analysis_features(self, features):
        '''Method to make features based on the trained survival models.'''
        # Get sociodemographics parameters of the patient
        sex = features['sex']
        age = features['actual_age']
        H = list(range(0, 24+1))

        # Make features based on the fitted Kaplan-Meier estimators (Males & Females)
        model = self.survival_models['kaplan_meier_both']
        s_age = model(age)
        for horizon in H:
            # Survival risk prediction S(t) at the t = AGE
            t = age + horizon/12.
            s = model(t)
            new_col = f'KaplanMeier_BOTH_S(AGE+{horizon}M)'
            features[new_col] = s

            # Difference |S(t+horizon) - S(t)|, where t = AGE
            ds = s - s_age
            new_col = f'KaplanMeier_BOTH_|S(AGE+{horizon}M)-S(AGE)|'
            features[new_col] = abs(ds)

        # Make sex-oriented features based on the fitted Kaplan-Meier estimators (Males and Females)
        model = self.survival_models['kaplan_meier_males'] if sex == 1 else self.survival_models['kaplan_meier_females']
        s_age = model(age)
        for horizon in H:
            # Survival risk prediction S(t) at the t = AGE
            t = age + horizon/12.
            s = model(t)
            new_col = f'KaplanMeier_SEX_S(AGE+{horizon}M)'
            features[new_col] = s

            # Difference |S(t+horizon) - S(t)|, where t = AGE
            ds = s - s_age
            new_col = f'KaplanMeier_SEX_|S(AGE+{horizon}M)-S(AGE)|'
            features[new_col] = abs(ds)

        # Make features based on the trained AFT model
        model = self.survival_models['aft']['model']
        covariates = self.survival_models['aft']['covariates']

        df = pd.DataFrame([{key:features[key] for key in covariates}])
        times = [age + val/12. for val in H]
        survivals = model.predict_survival_function(df, times=times)

        s_age = survivals[0].iloc[0]
        for i in range(len(survivals)):
            # Survival risk prediction S(t) at the t = AGE
            horizon = H[i]
            s = survivals[0].iloc[i]
            new_col = f'AFT_S(AGE+{horizon}M)'
            features[new_col] = s

            # Difference |S(t+horizon) - S(t)|, where t = AGE
            ds = s - s_age
            new_col = f'AFT_|S(AGE+{horizon}M)-S(AGE)|'
            features[new_col] = abs(ds)

        return features

    def feature_engineering(self,
            sex:        str,            # 'F' - FEMALE, 'M' - MALE
            birth_date: str,            # 'YYYY-MM-DD'
            ehr:        pd.DataFrame,   # Example is located in './EHR/id_26.csv'
            date_pred:  str,            # 'YYYY-MM-DD'
            deep_weeks: int             # > 0
    ):
        '''
            Method to make feature engineering for the Can-Save method.
            sex:              str      # 'F' - FEMALE, 'M' - MALE
            birth_date:       str      # 'YYYY-MM-DD'
            ehr:         pd.DataFrame  # Example is located in './EHR/id_26.csv'
            date_pred:        str      # 'YYYY-MM-DD'
            deep_weeks:       int      # deep_weeks > 0
        '''
        # Check the presence of the required columns in the EHR
        set_of_required_columns = {'date', 'code', 'is_diagnose'}
        missing_columns = set_of_required_columns.difference(set(ehr.columns))
        if len(missing_columns) > 0:
            raise KeyError(f'There are missing columns: {missing_columns}.')

        # Prepare dates
        ehr['date'] = pd.to_datetime(ehr['date'], format='%Y-%m-%d')
        date_pred = pd.to_datetime([date_pred], format='%Y-%m-%d')[0]
        date_start = date_pred - pd.DateOffset(weeks=deep_weeks)

        # Get sociodemographic parameters
        sex = 1 if 'M' in sex else 0
        birth_date = pd.to_datetime([birth_date], format='%Y-%m-%d')[0]

        # Select medical events in the certain period
        mask = (date_start <= ehr['date']) & (ehr['date'] <= date_pred)
        ehr = ehr[mask]

        # if EHR is empty, then we add a special medical service (placeholder) => EHR is always not empty
        if len(ehr) == 0:
            ehr.loc[0] = {'date': date_pred, 'code': 'A00.00', 'is_diagnose': 0}

        # Form the list of features for risk estimation
        days_per_year = (365 * 3 + 366) / 4.
        features = {
            'sex': sex,                                                     # SEX (0 - FEMALE, 1 - MALE)
            'actual_age': (date_pred - birth_date).days / days_per_year,    # ACTUAL AGE
        }

        # Make common features
        features = self.__common_features(features, ehr, date_start, date_pred)

        # Make features adopted to some regional specific characteristics
        features = self.__regional_features(features, ehr, date_start, date_pred)

        # Make features based on the trained survival models
        features = self.__survival_analysis_features(features)

        return features

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
