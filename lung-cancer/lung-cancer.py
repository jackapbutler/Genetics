# Data from https://www.kaggle.com/josemauricioneuro/lung-cancer-patients-mrna-microarray
DATA_PATH = "./data/complete_dataframe.csv"


import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import utils

df = pd.read_csv(DATA_PATH)
df = df.dropna(thresh=int(0.05 * len(df)))
df = df.drop(
    [
        "BlindedIDs",
        "PATIENT_ID",
        "Unnamed: 0",
        "WARNING",
        "DC_STUDY_ID",
        "LABORATORY_BATCH",
        "MONTHS_TO_LAST_CONTACT_OR_DEATH",
        "target",
        "MTHS_TO_LAST_CLINICAL_ASSESSMENT",
        "MONTHS_TO_FIRST_PROGRESSION",
    ],
    axis=1,
)

PATIENT_COLS = [
    "Stratagene",
    "SITE",
    "TESTTYPE",
    "IN_DC_STUDY",
    "GENDER",
    "AGE_AT_DIAGNOSIS",
    "RACE",
    "ADJUVANT_CHEMO",
    "ADJUVANT_RT",
    "VITAL_STATUS",
    "FIRST_PROGRESSION_OR_RELAPSE",
    "MONTHS_TO_FIRST_PROGRESSION",
    "MTHS_TO_LAST_CLINICAL_ASSESSMENT",
    "MONTHS_TO_LAST_CONTACT_OR_DEATH",
    "SMOKING_HISTORY",
    "SURGICAL_MARGINS",
    "PATHOLOGIC_N_STAGE",
    "PATHOLOGIC_T_STAGE",
    "MEDIAN_INTENSITY_UNNORMALIZED",
    "PCT_ARRAY_OUTLIER",
    "PCT_SINGLE_OUTLIER",
    "LABORATORY_BATCH",
    "Histologic grade",
]
TARGET_COL = "High_risk"  # look at months since dead/contact for more information
GENE_COLS = list(set(list(df.columns)) - set(PATIENT_COLS) - set([TARGET_COL]))

df = utils.one_hot_encode(df)
df = utils.cast_to_floats(df)

utils.missing_values_report(df)

y = df.pop(TARGET_COL)
df = StandardScaler().fit_transform(df)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.20)

log = LogisticRegression(C=0.05, solver="lbfgs", max_iter=500)
svm = SVC(probability=True)
knn = KNeighborsClassifier(n_neighbors=10)
adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=0.01)

LABELS = ["High risk", " Low risk"]

for model, name in zip(
    [log, svm, adaboost, knn], ["Logistic Regression", "SVM", "AdaBoost", "KNN"]
):
    utils.model_algorithm(model, X_train, y_train, X_test, y_test, name, LABELS)
