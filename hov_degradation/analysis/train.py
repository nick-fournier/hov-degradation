"""Runner script for training models"""
import numpy as np
import pandas as pd
import os
import json

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def check_path(path):
    if path[-1] is not "/":
        return path + "/"
    else:
        return path


def check_before_reading(path):
    with open(path) as f:
        first_line = f.readline()

    if "," in first_line:
        return pd.read_csv(path)

    if "\t" in first_line:
        return pd.read_csv(path, sep="\t")

class Detection:

    def __init__(self, inpath, outpath, date_range_string):
        # load data
        self.dates = date_range_string
        self.outpath = check_path(outpath)
        self.inpath = check_path(inpath)

        self.hyperparam_path = self.outpath + 'analysis/analysis_hyperparameters_' + self.dates + '.json'

        self.hyperparams = None
        self.scores = {}
        self.misconfig_ids = {}
        self.misconfig_meta = None

        # Read meta data
        self.flist = pd.Series(os.listdir(self.inpath))
        file = self.flist[self.flist.str.contains("meta")][0]
        self.df_meta = check_before_reading(self.inpath + file)

        self.district = str(self.df_meta.loc[0, 'District'])

        if not os.path.isdir(self.outpath):
            os.makedirs(self.outpath)
            os.makedirs(self.outpath + "analysis")

        with open(self.outpath + "processed/processed_neighbors_D" + self.district + "_" + self.dates + ".json") as f:
            self.neighbors = json.load(f)
        self.train_df_i210 = pd.read_csv(self.outpath + "processed/processed_i210_train_" + self.dates + ".csv", index_col=0).dropna()
        self.test_df_i210 = pd.read_csv(self.outpath + "processed/processed_i210_test_" + self.dates + ".csv", index_col=0).dropna()
        self.df_i210 = pd.concat([self.train_df_i210, self.test_df_i210], axis=0)
        self.df_District = pd.read_csv(self.outpath + "processed/processed_D" + self.district + "_" + self.dates + ".csv", index_col=0).dropna()


        # Running the machine learning
        self.train_classification()
        self.train_unsupervised()

        # Merge misconfig data with sensor meta data
        self.get_misconfig_meta()

        print("Completed training and testing of data for " + self.dates)

    def hyperparam_search(self, x, y, classifiers_map):
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
            print("Tuning hyperparameters of {}".format(name))
            clf = func()
            # grid search
            grid_search = GridSearchCV(clf, param_grid=param_grid[name])
            grid_search.fit(x, y)
            hyperparams[name] = grid_search.best_params_
            print(hyperparams)

        return hyperparams

    def train_classification(self):
        # i210 train data
        x_train_i210 = self.train_df_i210.drop(columns=['Type', 'y']).values
        y_train_i210 = self.train_df_i210['y'].values

        # i210 test data
        x_test_i210 = self.test_df_i210.drop(columns=['Type', 'y']).values
        y_test_i210 = self.test_df_i210['y'].values

        # 5min
        x_District = self.df_District.drop(columns=['Type', 'y']).values

        # classifiers
        classifiers_map = {
            'KNN': KNeighborsClassifier,
            'Logistic Regression': LogisticRegression,
            'Decision Tree': DecisionTreeClassifier,
            'Random Forest': RandomForestClassifier,
            # 'SVM': SVC
        }

        # hyperparams
        if os.path.isfile(self.hyperparam_path):
            with open(self.hyperparam_path) as f:
                self.hyperparams = json.load(f)
        else:
            self.hyperparams = self.hyperparam_search(x=x_train_i210,    # FIXME yf validation data
                                                      y=y_train_i210,    # FIXME yf validation data
                                                      classifiers_map=classifiers_map)
            # dump hyperparams
            with open(self.hyperparam_path, 'w') as f:
                json.dump(self.hyperparams, f, sort_keys=True, indent=4)

        # train
        scores = {name: None for name in classifiers_map.keys()}
        np.random.seed(12345)
        for name, func in classifiers_map.items():
            # pass parameters of the classifiers based on the hyperparams
            clf = func(**self.hyperparams[name])
            clf.fit(x_train_i210, y_train_i210)
            train_score = clf.score(x_train_i210, y_train_i210)
            test_score = clf.score(x_test_i210, y_test_i210)
            scores[name] = {'train': train_score, 'test': test_score}

        # select the best model
        # best_model = max(scores, key=operator.itemgetter(1))
        best_model = 'Random Forest'  # TODO
        clf = classifiers_map[best_model](**self.hyperparams[best_model])
        clf.fit(x_train_i210, y_train_i210)

        # predict the ids onto the processed data
        y_pred_District = clf.predict(x_District)
        self.df_District['preds_classification'] = y_pred_District

        # identify misconfigured ids
        misconfig_ids = list(self.df_District[self.df_District['preds_classification'] == 1].index)
        print("Anomalies detected by the classification model: "
              "{}".format(misconfig_ids))

        # Store analysis to class objects, save to disk later
        self.misconfig_ids['classification'] = misconfig_ids
        self.scores['classification'] = scores

    def train_unsupervised(self):
        # processed data
        x_i210 = self.df_i210.drop(columns=['Type', 'y']).values
        y_i210 = self.df_i210['y'].values

        x_District = self.df_District.drop(columns=['Type', 'y']).values
        # x_District = df_District.drop(columns=['Type']).values

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

        # select the best model
        best_model = max(scores, key=scores.get)
        func = unsupervised_map[best_model]
        # func.fit(x_District)
        # fit the data and tag outliers
        if best_model == "Local Outlier Factor":
            y_pred_District = func.fit_predict(x_District)
        else:
            y_pred_District = func.fit(x_District).predict(x_District)

        # change output labels for consistency with the classification outputs
        y_pred_District[y_pred_District == 1] = 0
        y_pred_District[y_pred_District == -1] = 1

        # add predictions to the dataframe
        self.df_District['preds_unsupervised'] = y_pred_District

        # identify misconfigured ids
        misconfig_ids = list(self.df_District[self.df_District['preds_unsupervised'] == 1].index)
        print("Anomalies detected by the unsupervised model: "
              "{}".format(misconfig_ids))

        # Store analysis to class objects, save to disk later
        self.misconfig_ids['unsupervised'] = misconfig_ids
        self.scores['unsupervised'] = scores

    def get_misconfig_meta(self):
        # Save misconfig IDs
        cols = ['ID', 'Fwy', 'Dir', 'District', 'Abs_PM', 'Length', 'Type', 'Lanes', 'Name', 'Latitude', 'Longitude']

        # Create nice data frame
        df_mis_ids = pd.DataFrame({'id': pd.Series(sum(self.misconfig_ids.values(), [])).unique(),
                                   'classification': False,
                                   'unsupervised': False})

        df_mis_ids = df_mis_ids.merge(self.df_meta[self.df_meta['ID'].isin(df_mis_ids['id'])], left_on='id', right_on='ID')
        df_mis_ids = df_mis_ids[cols + ['classification', 'unsupervised']]

        for m in ['classification', 'unsupervised']:
            df_mis_ids.loc[df_mis_ids['ID'].isin(self.misconfig_ids[m]), m] = True

        self.misconfig_meta = df_mis_ids.sort_values('ID')

    def save(self):
        # store dataframe
        self.df_District.to_csv(self.outpath + "analysis/analysis_detections_table_D" + self.district + "_" + self.dates + ".csv")
        self.misconfig_meta.to_csv(self.outpath + "analysis/analysis_misconfigs_meta_table_D" + self.district + "_" + self.dates + ".csv")

        # Save scores
        with open(self.outpath + 'analysis/analysis_scores_D' + self.district + "_" + self.dates + '.json', 'w') as f:
            json.dump(self.scores, f, sort_keys=True, indent=4)

        # Save scores
        with open(self.outpath + 'analysis/analysis_misconfigs_ids_D' + self.district + "_" + self.dates + '.json', 'w') as f:
            json.dump(self.misconfig_ids, f, sort_keys=True, indent=4)

