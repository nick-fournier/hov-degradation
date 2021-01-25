import os
import pandas as pd

class GetDegradation:

    def __init__(self, path, bad_sensors, peak_hours):
        self.path = path
        self.meta = pd.read_csv(path + "meta_2020-11-16.csv")
        self.bad_ids = bad_sensors
        self.peak_am = peak_hours['peak_am']
        self.peak_pm = peak_hours['peak_pm']
        self.neighbors = None
        self.data = None
        self.degraded = None

    def Repeat(x):
        _size = len(x)
        repeated = []
        for i in range(_size):
            k = i + 1
            for j in range(k, _size):
                if x[i] == x[j] and x[i] not in repeated:
                    repeated.append(x[i])
        return repeated

    def get_neighbors(self):
        # get freeway names
        df_group_id = self.meta.groupby('ID').first()
        df_bad_id = df_group_id[df_group_id.index.isin(self.bad_ids['ID'])]

        neighbors_id = []
        for i, _id in enumerate(df_bad_id.index):
            f1 = df_group_id['Fwy'] == df_bad_id['Fwy'].loc[_id]
            f2 = df_group_id['Dir'] == df_bad_id['Dir'].loc[_id]
            f3 = df_group_id['Type'] == 'ML' #df_bad_id['Type'].loc[_id]
            f4 = df_group_id['Abs_PM'] == df_bad_id['Abs_PM'].loc[_id]
            neighbors_id.append(df_group_id[f1 & f2 & f3 & f4].index[0])

        df_neighbors = pd.DataFrame({'bad_HOV': df_bad_id.index.to_list(),
                                     'neighbor_ML': neighbors_id})
        df_neighbors = df_neighbors.merge(self.bad_ids, left_on='bad_HOV', right_on='ID').drop(columns='ID')

        self.neighbors = df_neighbors

        return df_neighbors

    def get_sensor_data(self):
        # Header
        headers = pd.read_csv(self.path + "hourly_headers.csv", index_col=0, header=0)

        #List of all useful IDs
        id_list = self.neighbors['bad_HOV'].to_list() + self.neighbors['neighbor_ML'].to_list()

        # Read files
        misconfigs = []
        gzlist = pd.Series(os.listdir(self.path))
        gzlist = gzlist[gzlist.str.contains("txt.gz")]

        for gzf in gzlist:
            # Read file
            df_hourly = pd.read_csv(self.path + gzf, header=None, names=headers)

            # Filter for misconfig'd sensors and peak hours only 6-9am and 3-6pm
            f_sensors = df_hourly['Station'].isin(id_list)
            f_time = df_hourly['Timestamp'].str.slice(11, 19).isin(self.peak_am + self.peak_pm)
            df_hourly = df_hourly[f_sensors & f_time]

            # Add peak hour and day indicator for aggregation later
            df_hourly['DATE'] = df_hourly['Timestamp'].str.slice(0, 10)
            df_hourly['PEAK'] = 'AM'
            df_hourly.loc[df_hourly['Timestamp'].str.slice(11, 19).isin(self.peak_pm), 'PEAK'] = 'PM'

            # Add to output list
            misconfigs.append(df_hourly)
            print("Done loadings " + gzf)

        # Bind together
        df_peakhour = pd.concat(misconfigs)

        # Aggregation groups
        bool = df_peakhour.columns.isin(
            ['DATE', 'PEAK', 'Station', 'District', 'Freeway', 'Direction', 'Lane Type', 'Station Length']
        )
        agg_cols = df_peakhour.columns[bool].to_list()

        # Aggregate
        df_aggpeakhour = df_peakhour.groupby(agg_cols).agg('mean').reset_index()

        self.data = df_aggpeakhour

        return df_aggpeakhour

    def get_degradation(self):
        _cols = ['Station', 'District', 'Freeway', 'Direction', 'Lane Type',
                 'Station Length', 'DATE', 'PEAK', 'Samples', 'Observed']
        bad_cols = _cols + ['Flow', 'Occupancy', 'Speed']

        hov_results = []
        for bad_id in self.bad_ids['ID']:
            good_id = int(self.neighbors[self.neighbors['bad_HOV']==bad_id]['neighbor_ML'])
            good_lane = self.neighbors[self.neighbors['bad_HOV'] == bad_id]['real_lane'].to_string(index=False).strip()
            good_cols = _cols + [good_lane + s for s in [' Flow', ' Occupancy', ' Speed']]

            good_data = self.data[self.data['Station'] == good_id][good_cols]
            bad_data = self.data[self.data['Station'] == bad_id][bad_cols]

            items = {
                'Erroneous HOV': bad_id,
                'Correct from ML': good_id,
                'Correct ML lane': good_lane,
                'Erroneous Avg Speed': bad_data['Speed'].agg('mean'),
                'Corrected Avg Speed': good_data[good_lane + ' Speed'].agg('mean'),
                'Erroneous % of Days Speed < 45 mph': 100 * sum(bad_data['Speed'] < 45) / len(bad_data),
                'Corrected % of Days Speed < 45 mph': 100 * sum(good_data[good_lane + ' Speed'] < 45) / len(good_data)
            }
            hov_results.append(pd.DataFrame([items]))

        df_results = pd.concat(hov_results)

        col_order = ['Erroneous HOV', 'Correct from ML', 'Correct ML lane',
                     'Erroneous Avg Speed', 'Corrected Avg Speed',
                     'Erroneous % of Days Speed < 45 mph', 'Corrected % of Days Speed < 45 mph']
        df_results = df_results[col_order]

        self.results = df_results

        return df_results


if __name__ == '__main__':
    #path = "../../experiments/raw data/D7/"
    inpath = "experiments/raw data/D7/hourly/"
    outpath = "experiments/district_7/results/"
    dates = '2020-12-06' + '_to_' + '2020-12-12'

    #Peak hours
    peak_hours = {'peak_am': ['06:00:00', '07:00:00', '08:00:00', '09:00:00'],
                  'peak_pm': ['15:00:00', '16:00:00', '17:00:00', '18:00:00']}

    #These are the bad IDs and suspected mislabeled lane
    #df_bad = pd.read_csv(path + "meta_2020-11-16.csv")
    df_bad = pd.DataFrame(
        {'ID': [717822, 718270, 718313, 762500, 762549, 768743, 769238, 769745, 774055],
         'issue': ['Misconfigured']*9,
         'real_lane': ['Lane 1', 'Lane 2', 'Lane 1', 'Lane 2', 'Lane 1', 'Lane 4', 'Lane 1', 'Lane 3', 'Lane 1']}
    )


    degraded = GetDegradation(inpath, df_bad, peak_hours)
    neighbors = degraded.get_neighbors()
    sensordata = degraded.get_sensor_data()
    results = degraded.get_degradation()

    results.to_csv(outpath + "degradation_results.csv")




