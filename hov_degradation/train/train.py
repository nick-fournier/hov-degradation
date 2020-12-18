"""Runner script for training models"""
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import operator
import datetime

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score

from hov_degradation.utils.plot import save_plots
from hov_degradation.models.classifiers.neural_net import FeedForwardClassifier


def hyperparam_search(x, y, classifiers_map):
    """
    """
    param_grid = {
        'KNN': {
            'n_neighbors': [*range(1, 10)]
        },
        'Decision Tree': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [*range(1, 10)],
        },
        'Random Forest': {
            'n_estimators': [*range(10, 210, 40)],
            'criterion': ['gini', 'entropy'],
            'max_depth': [*range(1, 10)],
            'max_features': [*range(1, 10)]
        },
        'Logistic Regression': {
            'penalty': ['l1', 'l2'],
            'C': np.logspace(-4, 4, 20)
        },
        'SVM': {
            'C': np.logspace(-5, 4, 20),  # stats.expon(scale=100)
            'kernel': ['linear', 'rbf'],
            'gamma': np.logspace(-6, 4, 20),  # stats.expon(scale=.1)
            'class_weight': ['balanced', None]
        }
    }

    hyperparams = {name: None for name in classifiers_map.keys()}

    for name, func in classifiers_map.items():
        print("Tunning hyperparameters of {}".format(name))
        clf = func()
        # grid search
        grid_search = GridSearchCV(clf,
                                   param_grid=param_grid[name])
        grid_search.fit(x, y)
        hyperparams[name] = grid_search.best_params_
        print(hyperparams)

    return hyperparams


def train_classification(train_df_i210,
                         test_df_i210,
                         df_D7,
                         df_date,
                         hyperparam_path=None):
    # i210 train data
    x_train_i210 = train_df_i210.drop(columns=['Type', 'y']).values
    y_train_i210 = train_df_i210['y'].values

    # i210 test data
    x_test_i210 = test_df_i210.drop(columns=['Type', 'y']).values
    y_test_i210 = test_df_i210['y'].values

    # D7
    x_D7 = df_D7.drop(columns=['Type', 'y']).values

    # classifiers
    classifiers_map = {
        'KNN': KNeighborsClassifier,
        'Logistic Regression': LogisticRegression,
        'Decision Tree': DecisionTreeClassifier,
        'Random Forest': RandomForestClassifier,
        # 'SVM': SVC
    }

    # hyperparams
    if hyperparam_path:
        with open(hyperparam_path) as f:
            hyperparams = json.load(f)
    else:
        hyperparams = hyperparam_search(x=x_train_i210,
                                        # FIXME yf validation data
                                        y=y_train_i210,
                                        # FIXME yf validation data
                                        classifiers_map=classifiers_map)
        # dump hyperparams
        with open('hyperparameters.json', 'w') as f:
            json.dump(hyperparams, f, sort_keys=True, indent=4)

    # train
    scores = {name: None for name in classifiers_map.keys()}
    np.random.seed(12345)
    for name, func in classifiers_map.items():
        # pass parameters of the classifiers based on the hyperparams
        clf = func(**hyperparams[name])

        clf.fit(x_train_i210, y_train_i210)
        train_score = clf.score(x_train_i210, y_train_i210)
        test_score = clf.score(x_test_i210, y_test_i210)

        scores[name] = {'train': train_score, 'test': test_score}

    # dump scores
    with open('scores_classification_' + df_date + '.json', 'w') as f:
        json.dump(scores, f, sort_keys=True, indent=4)
    print("Classification Scores: ", scores)

    # select the best model
    # best_model = max(scores, key=operator.itemgetter(1))
    best_model = 'Random Forest'  # TODO
    clf = classifiers_map[best_model](**hyperparams[best_model])
    clf.fit(x_train_i210, y_train_i210)

    # predict on D7
    y_pred_D7 = clf.predict(x_D7)
    df_D7['preds_classification'] = y_pred_D7

    # identify misconfigured ids
    misconfig_ids = list(df_D7[df_D7['preds_classification'] == 1].index)
    print("Anomalies detected by the classification model: "
          "{}".format(misconfig_ids))

    # store dataframe
    df_D7.to_csv(path + "predictions_D7_" + df_date + ".csv")

    return misconfig_ids


def train_unsupervised(df_D7, df_i210, df_date):
    # processed data
    x_i210 = df_i210.drop(columns=['Type', 'y']).values
    y_i210 = df_i210['y'].values

    x_D7 = df_D7.drop(columns=['Type', 'y', 'preds_classification']).values
    # x_D7 = df_D7.drop(columns=['Type']).values

    outliers_fraction = 0.09

    # define outlier/anomaly detection methods to be compared
    unsupervised_map = {
        "Robust covariance": EllipticEnvelope(contamination=outliers_fraction),
        "One-Class SVM": svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                         gamma=0.1),
        "Isolation Forest": IsolationForest(behaviour='new',
                                            contamination=outliers_fraction,
                                            random_state=42),
        "Local Outlier Factor": LocalOutlierFactor(
            n_neighbors=35, contamination=outliers_fraction)
    }

    # train
    scores = {name: None for name in unsupervised_map.keys()}
    np.random.seed(12345)
    for name, func in unsupervised_map.items():
        func.fit(x_i210)
        if name == "Local Outlier Factor":
            y_pred_i210 = func.fit_predict(x_i210)
        else:
            y_pred_i210 = func.fit(x_i210).predict(x_i210)

        # change output labels for consistency with the classification outputs
        y_pred_i210[y_pred_i210 == 1] = 0
        y_pred_i210[y_pred_i210 == -1] = 1

        # compute scores - only available for I-210 since we have ground truth
        scores[name] = accuracy_score(y_pred_i210, y_i210)

    # dump scores
    with open('scores_unsupervised_' + df_date + '.json', 'w') as f:
        json.dump(scores, f, sort_keys=True, indent=4)
    print("Unsupervised Scores: ", scores)

    # select the best model
    best_model = max(scores, key=scores.get)
    func = unsupervised_map[best_model]
    # func.fit(x_D7)
    # fit the data and tag outliers
    if best_model == "Local Outlier Factor":
        y_pred_D7 = func.fit_predict(x_D7)
    else:
        y_pred_D7 = func.fit(x_D7).predict(x_D7)

    # change output labels for consistency with the classification outputs
    y_pred_D7[y_pred_D7 == 1] = 0
    y_pred_D7[y_pred_D7 == -1] = 1

    # add predictions to the dataframe
    df_D7['preds_unsupervised'] = y_pred_D7

    # identify misconfigured ids
    misconfig_ids = list(df_D7[df_D7['preds_unsupervised'] == 1].index)
    print("Anomalies detected by the unsupervised model: "
          "{}".format(misconfig_ids))

    # store dataframe
    df_D7.to_csv(path + "predictions_D7_" + df_date + ".csv")

    return misconfig_ids


if __name__ == '__main__':
    # load processed data
    path = "../../experiments/district_7/"

    # path = "C:/git_clones/connected_corridors/hov-degradation/experiments/district_7/"

    # dates = pd.date_range("2020-05-24","2020-05-24")
    dates = pd.date_range('2020-10-25', '2020-10-31')

    for thedate in dates:
        # to datetime as string
        date = str(thedate.date())

        # load neighbors
        with open(path + "neighbors_D7_" + date + ".json") as f:
            neighbors = json.load(f)

        df_data = pd.read_csv(path + "data/station_5min_" + date + ".csv")
        df_meta = pd.read_csv(path + "data/meta_2020-11-16.csv")
        # df_meta = pd.read_csv(path + "data/meta_2020-05-23.csv")

        # load i-210 data - don't need train or test for unsupervised
        train_df_i210 = pd.read_csv(path + "processed_i210_train_" + date + ".csv", index_col=0)
        train_df_i210.dropna(inplace=True)
        test_df_i210 = pd.read_csv(path + "processed_i210_test_" + date + ".csv", index_col=0)
        test_df_i210.dropna(inplace=True)
        df_i210 = pd.concat([train_df_i210, test_df_i210], axis=0)

        # Load D7 data
        df_D7 = pd.read_csv(path + "processed_D7_" + date + ".csv", index_col=0)
        df_D7.dropna(inplace=True)

        # run classification models
        mis_ids_clf = train_classification(train_df_i210=train_df_i210,
                                           test_df_i210=test_df_i210,
                                           df_D7=df_D7,
                                           df_date=date,
                                           hyperparam_path='hyperparameters.json')

        # plot
        print("Saving classification plots for date "+ date)
        save_plots(df_data=df_data,
                   df_meta=df_meta,
                   neighbors=neighbors,
                   misconfig_ids=mis_ids_clf,
                   path=path + 'results/classification_' + date)

        # run unsupervised models
        df_D7 = pd.read_csv(path + "predictions_D7_" + date + ".csv", index_col=0)

        mis_ids_unsupervised = train_unsupervised(df_D7=df_D7,
                                                  df_i210=df_i210,
                                                  df_date=date)
        print("Saving unsupervised plots for date " + date)
        save_plots(df_data=df_data,
                   df_meta=df_meta,
                   neighbors=neighbors,
                   misconfig_ids=mis_ids_unsupervised,
                   path=path + 'results/unsupervised_' + date)

        # store misconfigured IDs
        common_ids = list(set(mis_ids_clf).intersection(mis_ids_unsupervised))
        uncommon_ids = list(set(mis_ids_clf).symmetric_difference(
            mis_ids_unsupervised))

        mis_ids = {'classification': mis_ids_clf,
                   'unsupervised': mis_ids_unsupervised,
                   'common IDs': common_ids,
                   'uncommon': uncommon_ids}

        # dump ids
        with open('misconfigured_ids_' + date + '.json', 'w') as f:
            json.dump(mis_ids, f, sort_keys=True, indent=4)

        print("Completed training and testing of data for " + date)