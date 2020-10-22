"""Runner script for training models"""
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score

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
        'Random Forrest': {
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


def main(train_df_i210, test_df_i210, df_D7, hyperparam_path=None):
    # load processed data
    path = "../../experiments/district_7/"

    # train data
    train_df_i210 = pd.read_csv(path + "processed_i210_train.csv", index_col=0)
    train_df_i210.dropna(inplace=True)
    x_train_i210 = train_df_i210.drop(columns=['Type', 'y']).values
    y_train_i210 = train_df_i210['y'].values

    # test data
    test_df_i210 = pd.read_csv(path + "processed_i210_test.csv", index_col=0)
    test_df_i210.dropna(inplace=True)
    x_test_i210 = test_df_i210.drop(columns=['Type', 'y']).values
    y_test_i210 = test_df_i210['y'].values

    # train data
    train_df_D7 = pd.read_csv(path + "processed_D7_train.csv", index_col=0)
    train_df_D7.dropna(inplace=True)
    x_train_D7 = train_df_D7.drop(columns=['Type']).values

    # test data
    test_df_D7 = pd.read_csv(path + "processed_D7_test.csv", index_col=0)
    test_df_D7.dropna(inplace=True)
    x_test_D7 = test_df_D7.drop(columns=['Type']).values

    # classifiers
    classifiers_map = {
        'KNN': KNeighborsClassifier,
        'Logistic Regression': LogisticRegression,
        'Decision Tree': DecisionTreeClassifier,
        'Random Forrest': RandomForestClassifier,
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

    # # predict on D7
    # preds_train_D7 = clf.predict(x_train_D7)  # FIXME clf generalize
    # train_df_D7['preds'] = preds_train_D7
    # preds_test_D7 = clf.predict(x_test_D7)
    # test_df_D7['preds'] = preds_test_D7
    #
    # misconfigs = []
    # train_mis = list(train_df_D7[train_df_D7['preds'] == 1].index)
    # misconfigs.append(train_mis)
    # test_mis = list(test_df_D7[test_df_D7['preds'] == 1].index)
    # misconfigs.append(test_mis)
    # num_mis = len(test_mis) + len(train_mis)
    # print("misconfigured ids: {}, num_miss: {} ".format(misconfigs, num_mis))
    #
    # train_df_D7.to_csv(path + "prdictions_D7_train.csv")
    # test_df_D7.to_csv(path + "prdictions_D7_test.csv")

    with open('scores.json', 'w') as f:
        json.dump(scores, f, sort_keys=True, indent=4)
    print(scores)


def train_unsupervised(data_D7, data_i210):
    # processed data
    x_i210 = data_i210.drop(columns=['Type', 'y']).values
    y_i210 = data_i210['y'].values

    # x_D7 = data_D7.drop(columns=['Type', 'y']).values
    x_D7 = data_D7.drop(columns=['Type']).values

    # Example settings  - copied from https://scikit-learn.org/0.20/auto_examples/plot_anomaly_comparison.html
    n_samples = 300
    outliers_fraction = 0.15
    n_outliers = int(outliers_fraction * n_samples)
    n_inliers = n_samples - n_outliers

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
    with open('scores_unsupervised.json', 'w') as f:
        json.dump(scores, f, sort_keys=True, indent=4)
    print(scores)

    # select the best model
    best_model = max(scores, key=scores.get)
    func = unsupervised_map[best_model]
    func.fit(x_D7)
    # fit the data and tag outliers
    if best_model == "Local Outlier Factor":
        y_pred_D7 = func.fit_predict(x_D7)
    else:
        y_pred_D7 = func.fit(x_D7).predict(x_D7)

    # change output labels for consistency with the classification outputs
    y_pred_D7[y_pred_D7 == 1] = 0
    y_pred_D7[y_pred_D7 == -1] = 1

    # add predictions to the dataframe
    data_D7['preds_unsupervised'] = y_pred_D7

    # identify misconfigured ids
    misconfig_ids = list(data_D7[data_D7['preds_unsupervised'] == 1].index)
    print("misconfigured ids: {}".format(misconfig_ids))

    # store dataframe
    data_D7.to_csv(path + "prdictions_D7.csv")

    import ipdb;
    ipdb.set_trace()
    return misconfig_ids


if __name__ == '__main__':
    # load processed data
    path = "../../experiments/district_7/"

    # load i-210 data - don't need train or test for unsupervised
    train_df_i210 = pd.read_csv(path + "processed_i210_train.csv", index_col=0)
    train_df_i210.dropna(inplace=True)
    test_df_i210 = pd.read_csv(path + "processed_i210_test.csv", index_col=0)
    test_df_i210.dropna(inplace=True)
    data_i210 = pd.concat([train_df_i210, test_df_i210], axis=0)

    # Load D7 data
    train_df_D7 = pd.read_csv(path + "processed_D7_train.csv", index_col=0)
    test_df_D7 = pd.read_csv(path + "processed_D7_test.csv", index_col=0)
    train_df_D7.dropna(inplace=True)
    test_df_D7.dropna(inplace=True)
    data_D7 = pd.concat([train_df_D7, test_df_D7], axis=0)

    misconfig_ids_unsupervised = train_unsupervised(data_D7, data_i210)
    misonfigured_ids_classification =
    # main(hyperparam_path='hyperparameters.json')
    # main()