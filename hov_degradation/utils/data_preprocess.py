"""Script to preprocess PeMS Station 5-Minute data"""
import pandas as pd


def preprocess(path):
    data = pd.read_csv(path)

    # change column names
    general_columns = ['Timestamp', 'Station', 'District', 'Freeway_Num',
                       'Direction', 'Lane_Type', 'Station_Length', 'Samples',
                       'Percent_Observed', 'Total_Flow', 'Avg_Occupancy',
                       'Avg_Speed']
    # columns related to the lanes. Each segment might have different number of
    # lanes
    lane_cols = []
    for i in range(data.shape[1] - len(general_columns)):
        lane_cols.append("lane_info_{}".format(i))

    columns = general_columns + lane_cols

    data.columns = columns

    # filter, only keep HOV and mainline
    data = data[(data.Lane_Type == 'ML') | (data.Lane_Type == 'HV')]

    return data


if __name__ == '__main__':
    path = "../../experiments/district_7/data/d07_text_station_5min_2020_01_01.txt"
    data = preprocess(path=path)
