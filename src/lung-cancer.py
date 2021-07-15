# Data from https://www.kaggle.com/josemauricioneuro/lung-cancer-patients-mrna-microarray 
DATA_PATH='data/lung-cancer-kaggle/complete_dataframe.csv'

import pandas as pd
df = pd.read_csv(DATA_PATH)
df = df.dropna(thresh=int(0.1*len(df)))
df = df.drop(['target', 'BlindedIDs', 'PATIENT_ID'], axis=1)
columns = list(df.columns)

print('Column Names \n', columns)
print('\n High Risk Counts \n', df['High_risk'].value_counts())

PATIENT_COLS = ['Stratagene', 'DC_STUDY_ID', 'SITE', 'TESTTYPE', 'IN_DC_STUDY', 'GENDER', 'AGE_AT_DIAGNOSIS', 'RACE', 'ADJUVANT_CHEMO', 'ADJUVANT_RT', 'VITAL_STATUS', 'FIRST_PROGRESSION_OR_RELAPSE', 'MONTHS_TO_FIRST_PROGRESSION', 'MTHS_TO_LAST_CLINICAL_ASSESSMENT', 'MONTHS_TO_LAST_CONTACT_OR_DEATH', 'SMOKING_HISTORY', 'SURGICAL_MARGINS', 'PATHOLOGIC_N_STAGE', 'PATHOLOGIC_T_STAGE', 'MEDIAN_INTENSITY_UNNORMALIZED', 'PCT_ARRAY_OUTLIER', 'PCT_SINGLE_OUTLIER', 'WARNING', 'LABORATORY_BATCH', 'Histologic grade']
TARGET_COL = 'High_risk'
GENE_COLS = list(set(columns) - set(PATIENT_COLS) - set([TARGET_COL]))
