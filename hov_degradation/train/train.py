"""Script for training models"""
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

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


def main(hyperparam_path=None):
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
        hyperparams = hyperparam_search(x=x_train_i210,  # FIXME yf validation data
                                        y=y_train_i210,  # FIXME yf validation data
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

    #
    # # train neural net
    # num_layers = 3
    # hidden_units = 128
    # activation_fn = tf.keras.layers.LeakyReLU()
    # epochs = 10000
    # batch_size = 10
    # learning_rate = 0.001
    # target_ckpt = 'checkpoints/mlp.ckpt'
    # MLP = FeedForwardClassifier(x=x_train,
    #                             y=y_train,
    #                             x_test=x_test,
    #                             y_test=y_test,
    #                             num_layers=num_layers,
    #                             hidden_units=hidden_units,
    #                             activation_fn=activation_fn,
    #                             epochs=epochs,
    #                             batch_size=batch_size,
    #                             learning_rate=learning_rate,
    #                             target_ckpt=target_ckpt)
    # model = MLP.build_model()
    # history = MLP.train()

    with open('scores.json', 'w') as f:
        json.dump(scores, f, sort_keys=True, indent=4)
    print(scores)


if __name__ == '__main__':
    main(hyperparam_path='hyperparameters.json')
    # main()
