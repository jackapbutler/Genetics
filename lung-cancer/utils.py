from typing import List

import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series

import wandb


def model_algorithm(clf, X_train: DataFrame, y_train: Series, X_test: DataFrame, y_test: Series, name: str, labels: List):
    """Train a given model on the training data and show classification performance on WandB"""
    print("\n Starting training model; ", clf)
    
    clf.fit(X_train, y_train)
    y_probas = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)

    wandb.init(project="Genetics Lung Cancer", name=name, reinit=True)
    wandb.termlog('\nPlotting %s.'%name)

    wandb.sklearn.plot_learning_curve(clf, X_train, y_train)
    wandb.termlog('Logged learning curve.')

    wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels)
    wandb.termlog('Logged confusion matrix.')

    wandb.sklearn.plot_summary_metrics(clf, X=X_train, y=y_train, X_test=X_test, y_test=y_test)
    wandb.termlog('Logged summary metrics.')

    wandb.sklearn.plot_class_proportions(y_train, y_test, labels)
    wandb.termlog('Logged class proportions.')
    
    wandb.sklearn.plot_calibration_curve(clf, X_train, y_train, name)
    wandb.termlog('Logged calibration curve.')

    wandb.sklearn.plot_roc(y_test, y_probas, labels)
    wandb.termlog('Logged roc curve.')

    wandb.sklearn.plot_precision_recall(y_test, y_probas, labels)
    wandb.termlog('Logged precision recall curve.')

def one_hot_encode(data: DataFrame) -> DataFrame:
    """Encodes any non-numerical columns via one-hot encoding"""
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    non_numeric: DataFrame = data.select_dtypes(exclude=numerics)

    one_hot_cols = list(non_numeric.columns)
    df = pd.get_dummies(data=data, columns=one_hot_cols, dtype=float)

    print("\n One-hot encoded columns; \n", one_hot_cols)
    print("Increased dimensions from ", len(data.columns), " to ", len(df.columns))
    return df

def cast_to_floats(df: DataFrame) -> DataFrame:
    """Cast any integer columns to float64"""
    non_float = df.select_dtypes(exclude=['float64'])
    cols = list(non_float.columns)

    for i in range(len(cols)):
        df[cols[i]].astype('float64')
    
    print("\n Columns converted to float64; \n ", cols)
    return df

def missing_values_report(df: DataFrame):
    """Create missing value report for a DataFrame"""
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'}
    )
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)
    
    print("\n There are " + str(mis_val_table_ren_columns.shape[0]) + 
    " columns that have missing values. \n", mis_val_table_ren_columns
    )
