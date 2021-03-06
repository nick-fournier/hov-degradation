"""
Copyright ©2021. The Regents of the University of California (Regents). All Rights Reserved. Permission to use, copy, modify, and distribute this software and its documentation for educational, research, and not-for-profit purposes, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright notice, this paragraph and the following two paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""

import numpy as np
import pandas as pd
import json
import os
from scipy.stats import ks_2samp

FREEWAYS = {
    210: {
        'name': 'I210',
        'dirs': ['E', 'W'],
        'range_min': 22.6,
        'range_max': 41.4
    },
    134: {
        'name': 'SR134',
        'dirs': ['E', 'W'],
        'range_min': 11.4,
        'range_max': 13.5
    },
    605: {
        'name': 'I605',
        'dirs': ['N', 'S'],
        'range_min': 25.3,
        'range_max': 28.0
    }
}

# Hardcoded from static files
headers = ['Timestamp', 'Station', 'District', 'Freeway', 'Direction', 'Lane Type',
           'Station Length', 'Samples', 'Observed', 'Flow', 'Occupancy', 'Speed',
           'Lane 1 Samples', 'Lane 1 Flow', 'Lane 1 Occupancy', 'Lane 1 Speed', 'Lane 1 Observed',
           'Lane 2 Samples', 'Lane 2 Flow', 'Lane 2 Occupancy', 'Lane 2 Speed', 'Lane 2 Observed',
           'Lane 3 Samples', 'Lane 3 Flow', 'Lane 3 Occupancy', 'Lane 3 Speed', 'Lane 3 Observed',
           'Lane 4 Samples', 'Lane 4 Flow', 'Lane 4 Occupancy', 'Lane 4 Speed', 'Lane 4 Observed',
           'Lane 5 Samples', 'Lane 5 Flow', 'Lane 5 Occupancy', 'Lane 5 Speed', 'Lane 5 Observed',
           'Lane 6 Samples', 'Lane 6 Flow', 'Lane 6 Occupancy', 'Lane 6 Speed', 'Lane 6 Observed',
           'Lane 7 Samples', 'Lane 7 Flow', 'Lane 7 Occupancy', 'Lane 7 Speed', 'Lane 7 Observed',
           'Lane 8 Samples', 'Lane 8 Flow', 'Lane 8 Occupancy', 'Lane 8 Speed', 'Lane 8 Observed']


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


class PreProcess:
    """Preprocess PeMS Station 5-minute data for HOV anomaly detection.

    Attributes
    ----------
    # TODO
    """

    def __init__(self, inpath, outpath):
        """
        # TODO
        """
        #Initialize for later
        self.df_merge = None
        self.processed_data = None

        #Amend string
        self.inpath = check_path(inpath)
        self.outpath = check_path(outpath)

        # Read meta data
        self.flist = pd.Series(os.listdir(self.inpath))
        file = self.flist[self.flist.str.contains("meta")][0]
        self.df_meta = check_before_reading(self.inpath + file)

        self.district = str(self.df_meta.loc[0, 'District'])

        #Get the actual data
        self.df_data = self.getdata()

        #Extract the dates
        self.start_date = pd.to_datetime(self.df_data.Timestamp.min()[:10], format='%m/%d/%Y').strftime('%Y-%m-%d')
        self.end_date = pd.to_datetime(self.df_data.Timestamp.max()[:10], format='%m/%d/%Y').strftime('%Y-%m-%d')
        self.dates = pd.date_range(self.start_date, self.end_date)

    def getdata(self):
        data_flist = self.flist[self.flist.str.contains("station_5min_")]
        # Checks whether it's running or in debug
        # if os.path.isfile('5min_headers.csv'):
        #     headers = pd.read_csv('5min_headers.csv', index_col=0, header=0).columns
        # elif os.path.isfile('static/5min_headers.csv'):
        #     headers = pd.read_csv('static/5min_headers.csv', index_col=0, header=0).columns
        # else:
        #     headers = pd.read_csv('hov_degradation/static/5min_headers.csv', index_col=0, header=0).columns

        df_data = pd.DataFrame()
        for f in data_flist:
            print('Loading data from file: ' + f)
            df_data = df_data.append(pd.read_csv(self.inpath + f, header=None), sort=False)
        df_data.columns = headers

        return df_data

    def preprocess(self, location='5min', split=True):
        """ #TODO yf

        Parameters
        ----------

        Return
        ------

        """
        # filter I-210, 134, and 605 stations
        if location == 'i210':
            df_meta_filtered = pd.DataFrame(columns=self.df_meta.columns)
            for fway, value in FREEWAYS.items():
                fway_meta = self.df_meta[(self.df_meta['Fwy'] == fway) &
                                         (self.df_meta['Abs_PM'] >= value['range_min']) &
                                         (self.df_meta['Abs_PM'] <= value['range_max'])]
                df_meta_filtered = pd.concat([df_meta_filtered, fway_meta],
                                             axis=0)
        else:
            df_meta_filtered = self.df_meta

        # filter, only keep HOV and mainline
        df_meta_filtered = df_meta_filtered[df_meta_filtered.Type.isin(['ML',
                                                                        'HV'])]

        # merge using ID
        self.df_merge = pd.merge(self.df_data, df_meta_filtered, how='inner',
                                 left_on='Station', right_on='ID')

        # filter by usable stations
        usable = self.usable_stations()
        self.df_merge = self.df_merge[
            self.df_merge.ID.isin(usable[usable].index)]

        # apply misconfiguration, swap Flow and Occupancy of some HOVs with MLs
        if location == 'i210':
            self.apply_misconfiguration(ratio=0.30)

        # get pivot based on Flow - Filter based on usable stations
        df_flow_piv = self.df_merge.pivot('Timestamp', 'ID', 'Flow')
        # df_flow_piv = df_flow_piv.loc[:, usable]

        # normalize flow based on the number of lanes
        df_group_id = self.df_merge.groupby('ID').first()
        df_flow_piv /= df_group_id['Lanes'][df_flow_piv.columns]

        # get pivot based on Occupancy - Filter based on usable stations
        df_ocupancy_piv = self.df_merge.pivot('Timestamp', 'ID', 'Occupancy')
        # df_ocupancy_piv = df_ocupancy_piv.loc[:, usable]

        # get nighttime average, for all days
        df_flow_piv.index = pd.to_datetime(df_flow_piv.index, format='%m/%d/%Y %H:%M:%S', errors='ignore')
        avg_nighttime_flow = df_flow_piv.between_time('01:00:00', '03:00:00').mean(axis=0)

        # begin_time = self.date + " 01:00:00" #"05/24/2020 01:00:00"  # datetime.datetime(2020,5,24,1,0,0)
        # end_time = self.date + " 03:00:00" #"05/24/2020 03:00:00"  # datetime.datetime(2020,5,24,3,0,0)
        # avg_nighttime_flow = df_flow_piv.T.loc[:, begin_time:end_time].mean(axis=1)

        # get K-S test value for downstream and upstream stations
        neighbors = self.get_neighbors(df_group_id)
        ks_stats_flow = self.get_ks_stats(df_flow_piv.T, neighbors)
        ks_stats_occupancy = self.get_ks_stats(df_ocupancy_piv.T, neighbors)

        # get IDs of the stations sorted by distance
        sorted_stations = self.df_merge.sort_values(
            by=['Fwy', 'Dir', 'Abs_PM']).ID.unique()

        # get target
        target = df_group_id[
            'misconfigured'] if location == 'i210' else -1

        self.processed_data = pd.DataFrame(
            index=sorted_stations,
            data={'avg_nighttime_flow': avg_nighttime_flow,
                  'ks_flow_up': ks_stats_flow['up']['p-value'],
                  'ks_flow_down': ks_stats_flow['down']['p-value'],
                  'ks_flow_ml_up': ks_stats_flow['main_up']['p-value'],
                  'ks_flow_ml_down': ks_stats_flow['main_down']['p-value'],
                  'ks_flow_ml': ks_stats_flow['main']['p-value'],
                  'ks_occupancy_up': ks_stats_occupancy['up']['p-value'],
                  'ks_occupancy_down': ks_stats_occupancy['down']['p-value'],
                  'ks_occupancy_ml_up': ks_stats_occupancy['main_up'][
                      'p-value'],
                  'ks_occupancy_ml_down': ks_stats_occupancy['main_down'][
                      'p-value'],
                  'ks_occupancy_ml': ks_stats_occupancy['main']['p-value'],
                  'Type': df_group_id['Type'],
                  'y': target
                  })
        # filter only HOV, for now
        self.processed_data = self.processed_data[
            self.processed_data.Type == 'HV']

        # split test and train
        if split:
            df_train, df_test = self.split_data(self.processed_data)
        else:
            df_train = self.processed_data
            df_test = None

        return df_train, df_test, neighbors

    def get_ks_stats(self, X, neighbors):
        """ #TODO yf

        Parameters
        ----------
        X :

        neighbors :

        Return
        ------
        ks_down :

        ks_up :


        """
        ks_down = pd.DataFrame(index=X.index, columns=['statistic', 'p-value'])
        ks_up = pd.DataFrame(index=X.index, columns=['statistic', 'p-value'])
        ks_ml_down = pd.DataFrame(index=X.index,
                                  columns=['statistic', 'p-value'])
        ks_ml_up = pd.DataFrame(index=X.index, columns=['statistic', 'p-value'])
        ks_ml = pd.DataFrame(index=X.index, columns=['statistic', 'p-value'])

        inds = X.index

        for ind in inds:
            try:
                X_up = X.loc[neighbors[ind]['up']]
                ks_up.loc[ind, :] = ks_2samp(X.loc[ind].values, X_up.values)
            except KeyError:
                pass
            try:
                X_down = X.loc[neighbors[ind]['down']]
                ks_down.loc[ind, :] = ks_2samp(X.loc[ind].values, X_down.values)
            except KeyError:
                pass
            try:
                X_ml_up = X.loc[neighbors[neighbors[ind]['up']]['main']]
                ks_ml_up.loc[ind, :] = ks_2samp(X.loc[ind].values,
                                                X_ml_up.values)
            except KeyError:
                pass
            try:
                X_ml_down = X.loc[neighbors[neighbors[ind]['down']]['main']]
                ks_ml_down.loc[ind, :] = ks_2samp(X.loc[ind].values,
                                                  X_ml_down.values)
            except KeyError:
                pass
            try:
                X_ml = X.loc[neighbors[ind]['main']]
                ks_ml.loc[ind, :] = ks_2samp(X.loc[ind].values, X_ml.values)
            except KeyError:
                pass

        ks_stats = {'up': ks_up,
                    'down': ks_down,
                    'main_up': ks_ml_up,
                    'main_down': ks_ml_down,
                    'main': ks_ml}
        return ks_stats

    def get_neighbors(self, df_group_id):
        """ #TODO yf

        Parameters
        ----------
        df_group_id :
            pandas dataframe groupby object

        Return
        ------
        neighbors : dict

        """
        # get freeway names
        fwys = df_group_id.Fwy.unique()

        # get directions
        dirs = df_group_id.Dir.unique()

        # get freeway types
        typs = df_group_id.Type.unique()

        neighbors = {}
        for fwy in fwys:
            for dir in dirs:
                for typ in typs:
                    # find sorted ids
                    _ids = df_group_id[
                        (df_group_id['Fwy'] == fwy) &
                        (df_group_id['Dir'] == dir) &
                        (df_group_id['Type'] == typ)].sort_values(by='Abs_PM'
                                                                  ).index
                    for i, _id in enumerate(_ids):
                        neighbors[_id] = {'up': None,
                                          'down': None,
                                          'main': None,
                                          'hov': None}
                        # set upstream neighbors
                        if i > 0:
                            neighbors[_id].update({'up': int(_ids[i - 1])})

                        # set downstream neighbors
                        if i < len(_ids) - 1:
                            neighbors[_id].update({'down': int(_ids[i + 1])})

                        # set mainline neighbor of the HOV at the same location
                        if typ == 'HV':
                            try:
                                main_neighbor_id = df_group_id[
                                    (df_group_id['Fwy'] == fwy) &
                                    (df_group_id['Type'] == 'ML') &
                                    (df_group_id['Abs_PM'] == df_group_id.loc[
                                        # FIXME set almost equal
                                        _id, 'Abs_PM'])].index[0]
                                neighbors[_id].update(
                                    {'main': int(main_neighbor_id)})
                            except IndexError:
                                pass

                        # set HOV neighbor of the mainline at the same location
                        if typ == 'ML':
                            try:
                                hov_neighbor_id = df_group_id[
                                    (df_group_id['Fwy'] == fwy) &
                                    (df_group_id['Type'] == 'HV') &
                                    (df_group_id['Abs_PM'] == df_group_id.loc[
                                        # FIXME set almost equal
                                        _id, 'Abs_PM'])].index[0]
                                neighbors[_id].update(
                                    {'hov': int(hov_neighbor_id)})
                            except IndexError:
                                pass
        return neighbors

    def usable_stations(self):
        """ #TODO yf

        Return
        ------
        usable :

        """
        group = self.df_merge.groupby('ID')
        num_times = len(self.df_merge['Timestamp'].unique())

        # Find IDS that report Flow for more than half of the timestamps
        fidx = group['Flow'].count() > (num_times / 2)

        # Find IDS that report at least 5 samples per lane
        sidx = group['Samples'].mean() / group.first()['Lanes'] > 5.0

        # Find IDs that report mean observation > 50%
        oidx = group['Observed'].mean() > 50

        usable = fidx & sidx & oidx
        return usable

    def apply_misconfiguration(self, ratio=0.30):
        """ #TODO yf

        Parameters
        ----------
        data :
            pandas dataframe

        Return
        ------
        data :

        """
        # create new columns for swapped data, initialize based on ground truth
        self.df_merge['misconfigured'] = 0
        self.df_merge['swapped_lane_num'] = None

        df_group_id = self.df_merge.groupby('ID').first()

        # filter
        hov_data = df_group_id[df_group_id.Type == 'HV']

        # randomly select stations
        np.random.seed(12345678)

        misconfigured_inds = np.random.randint(low=0,
                                               high=hov_data.shape[0],
                                               size=int(
                                                   hov_data.shape[0] * ratio))
        misconfigured_ids = hov_data.iloc[misconfigured_inds].index

        # assign label to misconfigured stations

        # get neighbors
        neighbors = self.get_neighbors(df_group_id)

        # swap hov info (Flow and Occupancies) with a random lane in mainline
        for _id in misconfigured_ids:
            self.df_merge.loc[self.df_merge.ID == _id, 'misconfigured'] = 1

            # get neighbor in mainline
            neighbor_ml_id = neighbors[_id]['main']

            # swap hov with a lane in mainline
            if neighbor_ml_id is not None:
                num_lanes = df_group_id['Lanes'][neighbor_ml_id]

                # get the lane number of the mainline to be swapped with
                if num_lanes > 1:
                    # get random lane
                    lane_num = np.random.randint(low=1,
                                                 high=num_lanes,
                                                 size=1)[0]
                else:
                    lane_num = 1

                self.df_merge.loc[
                    self.df_merge.ID == _id, 'swapped_lane_num'] = lane_num

                lane_cols = ['Lane {} Flow'.format(lane_num),
                             'Lane {} Occupancy'.format(lane_num)]
                total_cols = ['Flow', 'Occupancy']

                #Ensuring consistency (no missing time stamps between sensors)
                missing = []
                missing += list(set(self.df_merge.loc[self.df_merge.ID == neighbor_ml_id, 'Timestamp']).difference(
                    self.df_merge.loc[self.df_merge.ID == _id, 'Timestamp']))
                missing += list(set(self.df_merge.loc[self.df_merge.ID == _id, 'Timestamp']).difference(
                    self.df_merge.loc[self.df_merge.ID == neighbor_ml_id, 'Timestamp']))

                self.df_merge = self.df_merge.loc[~self.df_merge.Timestamp.isin(missing), ]

                # update total Flow of the mainline # TODO hov > 1 lanes, Fix else
                self.df_merge.loc[self.df_merge.ID == neighbor_ml_id, 'Flow'] = \
                    self.df_merge.loc[self.df_merge.ID == neighbor_ml_id, 'Flow'].values + \
                    self.df_merge.loc[self.df_merge.ID == _id, 'Flow'].values - \
                    self.df_merge.loc[self.df_merge.ID == neighbor_ml_id, lane_cols[0]].values


                # update total Occupancy of the mainline
                self.df_merge.loc[
                    self.df_merge.ID == neighbor_ml_id, 'Occupancy'] = \
                    self.df_merge.loc[
                        self.df_merge.ID == neighbor_ml_id, 'Occupancy'].values \
                    + (self.df_merge.loc[
                           self.df_merge.ID == _id, 'Occupancy'].values -
                       self.df_merge.loc[
                           self.df_merge.ID == neighbor_ml_id, lane_cols[
                               1]].values) / num_lanes

                # swap Flow of hov with the lane of mainline
                self.df_merge.loc[self.df_merge.ID == _id, 'Flow'], \
                self.df_merge.loc[
                    self.df_merge.ID == neighbor_ml_id, lane_cols[0]] = \
                    self.df_merge.loc[
                        self.df_merge.ID == neighbor_ml_id, lane_cols[
                            0]].values, \
                    self.df_merge.loc[self.df_merge.ID == _id, 'Flow'].values

                # swap Occupancy of hov with the lane of mainline
                self.df_merge.loc[self.df_merge.ID == _id, 'Occupancy'], \
                self.df_merge.loc[
                    self.df_merge.ID == neighbor_ml_id, lane_cols[1]] = \
                    self.df_merge.loc[
                        self.df_merge.ID == neighbor_ml_id, lane_cols[
                            1]].values, \
                    self.df_merge.loc[
                        self.df_merge.ID == _id, 'Occupancy'].values

    def split_data(self, data, test_size=0.3, shuffle=True):
        """Splits data to training, validation and testing parts

        Note: data is not shuffled to keep the time series for plotting
        acceleration profiles

        Parameters:
        ----------
        data : pandas.Dataframe
            pandas dataframe object of the preprocessed data
        test_size : float
            portion of the data used as test data
        shuffle : bool
            boolean, if true, will randomly shuffle the data

        Returns:
        -------
        df_train : pandas.Dataframe
            pandas dataframe object containing training dataset
        df_test : pandas.Dataframe
            pandas dataframe object containing test dataset
        """
        if shuffle:
            data.sample(frac=1)

        n = int(len(data) * (1 - test_size))
        df_train, df_test = data.iloc[:n], data.iloc[n:]
        return df_train, df_test

    def save(self):
        if not os.path.isdir(self.outpath):
            os.makedirs(self.outpath)
            os.makedirs(self.outpath + "processed data")

        print("Pre-processing I-210 test/train data...")
        # I210
        train_i210, test_i210, neighbors_i210 = self.preprocess(location='i210')

        # District 7
        print("Pre-processing District data...")
        df_District, _, neighbors_District = self.preprocess(location='5min', split=False)

        print("Saving analysis...")
        # I210
        train_i210.to_csv(self.outpath + "processed data/i210_train_data_" + self.start_date + "_to_" + self.end_date + ".csv")
        test_i210.to_csv(self.outpath + "processed data/i210_test_data_" + self.start_date + "_to_" + self.end_date + ".csv")

        # District
        # self.df_meta.to_csv(self.outpath + "processed data/processed_meta_D" + self.district + "_" + self.start_date + "_to_" + self.end_date + ".csv")
        df_District.to_csv(self.outpath + "processed data/D" + self.district + "_data_" + self.start_date + "_to_" + self.end_date + ".csv")
        with open(self.outpath + "processed data/D" + self.district + "_neighbors_" + self.start_date + "_to_" + self.end_date + ".json", 'w') as f:
            json.dump(neighbors_District, f, sort_keys=True, indent=4)
        print("Done")
