import numpy as np
import pandas as pd
import os
import json
from collections import Counter



#### These functions I don't need anymore I don't think... ###

def agg_scores(path, dates):
    scores = pd.DataFrame()
    for thedate in dates:
        date = str(thedate.date())
        # Load em
        new_classif = pd.read_json(path + 'scores_classification_' + date + ".json", orient='index')
        new_classif['Type'] = 'Classification'

        new_unsuper = pd.read_json(path + 'scores_unsupervised_' + date + ".json", typ='series')
        new_unsuper = pd.DataFrame({'train': np.nan, 'test': new_unsuper})
        new_unsuper['Type'] = 'Unsupervised'

        # Stack em
        new_scores = pd.concat([new_classif, new_unsuper], axis=0, sort=False)

        #Label em
        new_scores["Date"] = date
        new_scores["Model"] = new_scores.index

        # Stack em and Add em
        scores = scores.append(pd.DataFrame(data=new_scores), ignore_index=True)

    # Aggregate
    agg_scores_test = scores.groupby(['Model', 'Type']).agg({'test': ['mean', 'std']})
    agg_scores_test.columns = ['test_mean', 'test_std']

    agg_scores_train = scores.groupby(['Model','Type']).agg({'train': ['mean', 'std']})
    agg_scores_train.columns = ['train_mean', 'train_std']

    # Concat
    agg_scores = pd.concat([agg_scores_test, agg_scores_train], axis=1)

    # Sort
    agg_scores = agg_scores.sort_values(by=['test_mean', 'Type'], ascending=False)
    scores = scores.sort_values(by=['test', 'Type'], ascending=False)

    #Clean up
    agg_scores.reset_index(inplace=True, level='Type')

    # Save
    datestring = str(dates.min().date()) + "_to_" + str(dates.max().date())

    # scores.to_csv('scores_classification_' + datestring + '.csv')
    scores.to_csv(path + 'scores_classification_' + datestring + '.csv', index=False)
    agg_scores.to_csv(path + 'agg_scores_classification_' + datestring + '.csv', index=False)

    out = agg_scores.to_json(orient='index')
    parsed = json.loads(out)
    with open(path + 'agg_scores_classification_' + datestring + '.json', 'w') as f:
        json.dump(parsed, f, sort_keys=True, indent=4)


def agg_misconfigs(path, dates):
    misconfig_classif = []
    misconfig_unsuper = []
    for thedate in dates:
        date = str(thedate.date())
        with open(path + 'misconfigured_ids_' + date + '.json') as f:
            new_misconfig = json.load(f)

        misconfig_classif.extend(new_misconfig['classification'])
        misconfig_unsuper.extend(new_misconfig['unsupervised'])

    # Count the occurances
    misconfig_classif = pd.DataFrame.from_dict(Counter(misconfig_classif), orient='index').reset_index()
    misconfig_unsuper = pd.DataFrame.from_dict(Counter(misconfig_unsuper), orient='index').reset_index()

    misconfig_classif.columns = ['ID', 'Classification']
    misconfig_unsuper.columns = ['ID', 'Unsupervised']

    # Merge together
    misconfig_ids = misconfig_classif.merge(misconfig_unsuper, left_on='ID', right_on='ID', how='outer').fillna(0)

    # Save
    datestring = str(dates.min().date()) + "_to_" + str(dates.max().date())
    misconfig_ids.to_csv(path + 'misconfigured_ids_frequency' + datestring + '.csv', index=False)
