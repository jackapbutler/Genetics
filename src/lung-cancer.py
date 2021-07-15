# Data from https://www.kaggle.com/josemauricioneuro/lung-cancer-patients-mrna-microarray 
DATA_PATH='data/lung-cancer-kaggle/complete_dataframe.csv'

import pandas as pd
df = pd.read_csv(DATA_PATH)
df = df.dropna(thresh=int(0.1*len(df)))
df = df.drop(['target', 'BlindedIDs', 'PATIENT_ID'], axis=1)

print('Column Names \n', list(df.columns))
print('\n High Risk Counts \n', df['High_risk'].value_counts())

