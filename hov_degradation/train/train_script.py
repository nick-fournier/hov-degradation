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
    train_df = pd.read_csv(path + "processed_train.csv")
    train_df.dropna(inplace=True)
    x_train = train_df.drop(columns=['Type', 'y']).values
    y_train = train_df['y'].values

    # test data
    test_df = pd.read_csv(path + "processed_test.csv")
    test_df.dropna(inplace=True)
    x_test = test_df.drop(columns=['Type', 'y']).values
    y_test = test_df['y'].values

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
        hyperparams = hyperparam_search(x=x_train,  # FIXME yf validation data
                                        y=y_train,  # FIXME yf validation data
                                        classifiers_map=classifiers_map)
        # dump hyperparams
        with open('hyperparameters.json', 'w') as f:
            json.dump(hyperparams, f, sort_keys=True, indent=4)

    # # train
    # scores = {name: None for name in classifiers_map.keys()}
    # np.random.seed(12345)
    # for name, func in classifiers_map.items():
    #     # pass parameters of the classifiers based on the hyperparams
    #     clf = func(**hyperparams[name])
    #
    #     clf.fit(x_train, y_train)
    #     train_score = clf.score(x_train, y_train)
    #     test_score = clf.score(x_test, y_test)
    #     scores[name] = {'train': train_score, 'test': test_score}

    # train neural net
    num_layers = 3
    hidden_units = 128
    activation_fn = tf.keras.layers.LeakyReLU()
    epochs = 10000
    batch_size = 10
    learning_rate = 0.001
    target_ckpt = 'checkpoints/mlp.ckpt'
    MLP = FeedForwardClassifier(x=x_train,
                                y=y_train,
                                x_test=x_test,
                                y_test=y_test,
                                num_layers=num_layers,
                                hidden_units=hidden_units,
                                activation_fn=activation_fn,
                                epochs=epochs,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                target_ckpt=target_ckpt)
    model = MLP.build_model()
    history = MLP.train()

    with open('scores.json', 'w') as f:
        json.dump(scores, f, sort_keys=True, indent=4)
    print(scores)


if __name__ == '__main__':
    main(hyperparam_path='hyperparameters.json')
    # main()