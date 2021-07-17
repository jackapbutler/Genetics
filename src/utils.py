from pandas.core.frame import DataFrame
import wandb
import pandas as pd

def model_algorithm(clf, X_train, y_train, X_test, y_test, name, labels):
    """Train a given model on the training data and show classification performance on WandB"""

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

def one_hot_encode(data: DataFrame):
    """Encodes any non-numerical columns via one-hot encoding"""
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    non_numeric = data.select_dtypes(exclude=numerics)

    one_hot_cols = list(non_numeric.columns)
    df = pd.get_dummies(data=data, columns=one_hot_cols, dtype=float)

    print("One-hot encoded columns; ", one_hot_cols)
    print("Increased dimensions from ", len(data.columns), " to ", len(df.columns))
    print(df.columns)
    return df