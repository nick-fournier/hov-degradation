import os
import pandas as pd

def get_neighbors(meta_data, bad_ids):
    # get freeway names
    df_group_id = meta_data.groupby('ID').first()
    df_bad_id = df_group_id[df_group_id.index.isin(bad_ids['ID'])]

    neighbors_id = []
    for i, _id in enumerate(df_bad_id.index):
        f1 = df_group_id['Fwy'] == df_bad_id['Fwy'].loc[_id]
        f2 = df_group_id['Dir'] == df_bad_id['Dir'].loc[_id]
        f3 = df_group_id['Type'] == 'ML' #df_bad_id['Type'].loc[_id]
        f4 = df_group_id['Abs_PM'] == df_bad_id['Abs_PM'].loc[_id]
        neighbors_id.append(df_group_id[f1 & f2 & f3 & f4].index[0])

    df_neighbors = pd.DataFrame({'bad_HOV': df_bad_id.index.to_list(),
                                 'neighbor_ML': neighbors_id})

    return df_neighbors


def get_degradation(path, neighbors, peak_am, peak_pm):
    # Header
    headers = pd.read_csv(path + "hourly_headers.csv", index_col=0, header=0)

    #List of all useful IDs
    id_list = neighbors.iloc[:, 0].to_list() + neighbors.iloc[:, 1].to_list()

    # Read files
    misconfigs = []
    gzlist = pd.Series(os.listdir(path))
    gzlist = gzlist[gzlist.str.contains("txt.gz")]

    for gzf in gzlist:
        # Read file
        df_hourly = pd.read_csv(path + gzf, header=None, names=headers)

        # Filter for misconfig'd sensors and peak hours only 6-9am and 3-6pm
        f_sensors = df_hourly['Station'].isin(id_list)
        f_time = df_hourly['Timestamp'].str.slice(11, 19).isin(peak_am + peak_pm)
        df_hourly = df_hourly[f_sensors & f_time]

        # Add peak hour and day indicator for aggregation later
        df_hourly['DATE'] = df_hourly['Timestamp'].str.slice(0, 10)
        df_hourly['TIME'] = 'AM'
        df_hourly.loc[df_hourly['Timestamp'].str.slice(11, 19).isin(peak_pm), 'TIME'] = 'PM'

        # Add to output list
        misconfigs.append(df_hourly)
        print("Done with " + gzf)

    # Bind together
    df_peakhour = pd.concat(misconfigs)

    # Aggregate
    agg_cols = ['DATE', 'TIME', 'Station', 'District', 'Freeway', 'Direction', 'Lane Type', 'Station Length']

    df_peakhour = df_peakhour.groupby(agg_cols).agg('mean').reset_index()

    return df_peakhour



if __name__ == '__main__':
    #path = "../../experiments/raw data/D7/"
    path = "experiments/raw data/D7/hourly/"
    start_date = '2020-12-06'
    end_date = '2020-12-12'
    dates = start_date + '_to_' + end_date

    #These are the bad IDs and suspected mislabeled lane
    #df_bad = pd.read_csv(path + "meta_2020-11-16.csv")
    df_bad = pd.DataFrame(
        {'ID': [717822, 718270, 718313, 762500, 762549, 768743, 769238, 769745, 774055],
         'issue': ['Misconfigured']*9,
         'real_lane': ['Lane 1', 'Lane 2', 'Lane 1', 'Lane 2', 'Lane 1', 'Lane 4', 'Lane 1', 'Lane 3', 'Lane 1']}
    )
    #IDs of corresponding mainlines
    neighbors = get_neighbors(df_meta, df_bad)

    # Sensor meta data
    df_meta = pd.read_csv(path + "meta_2020-11-16.csv")

    #Peak hours
    peak_am = ['06:00:00', '07:00:00', '08:00:00', '09:00:00']
    peak_pm = ['15:00:00', '16:00:00', '17:00:00', '18:00:00']

    df_misconfigs = extract_sensor_data(path)




