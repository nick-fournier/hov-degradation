import os
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

class GetDegradation:

    def __init__(self, path, bad_sensors, peak_hours, saved=False, joedays=False, metaname='d07_text_meta_2019_11_09.txt'):
        self.path = path
        self.meta = pd.read_csv(path + metaname)
        self.bad_ids = bad_sensors
        self.peak_am = peak_hours['peak_am']
        self.peak_pm = peak_hours['peak_pm']
        self.joedays = joedays
        self.neighbors = None
        self.raw_data = None
        self.data = None
        self.degraded = None
        self.saved = saved

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

        # self.neighbors = df_neighbors
        return df_neighbors

    def get_sensor_data(self):

        fpath = self.path + 'd07_extracted_sensors_hourly.csv'

        if self.saved and os.path.isfile(fpath):
                df_sensors = pd.read_csv(fpath)
        else:
            # Headers, file list, and ID list
            headers = pd.read_csv(self.path + "d07_hourly_headers.csv", index_col=0, header=0)
            id_list = self.neighbors['bad_HOV'].to_list() + self.neighbors['neighbor_ML'].to_list()
            misconfigs = []
            gzlist = pd.Series(os.listdir(self.path))
            gzlist = gzlist[gzlist.str.contains("txt.gz")]

            # Read file, Filter for misconfig'd sensors
            for gzf in gzlist:
                df_hourly = pd.read_csv(self.path + gzf, header=None, names=headers)
                df_hourly = df_hourly[df_hourly['Station'].isin(id_list)]
                misconfigs.append(df_hourly)
                print("Done loadings " + gzf)

            # Bind together &  Rename Station length to get rid of space
            df_sensors = pd.concat(misconfigs)
            df_sensors.rename(columns={"Station Length": "Length"}, inplace=True)
            #Save
            df_sensors.to_csv(fpath)

        # self.raw_data = df_sensors
        return df_sensors

    def get_vhtvmt(self, df, suffix):
        # Calculate VMT and VHT
        df = df.assign(VMT=df.eval('Flow * Length'))
        df = df.assign(VHT=df.eval('(Flow * Length) / Speed'))

        # Aggregate & remove <100 observable
        grp_cols = df.columns[~df.columns.isin(['VMT', 'VHT', 'Flow', 'Occupancy', 'Speed'])].to_list()
        df = df.groupby(grp_cols).agg({'VMT': 'sum', 'VHT': 'sum'}).reset_index()
        df = df[df['Observed'] == 100]

        # Calculate speed from VMT/VHT
        df['Avg Speed'] = df.eval('VMT / VHT')

        #Calculate the sum totals
        vmt = df['VMT'].sum()
        vht = df['VHT'].sum()
        spd = vmt / vht
        ndays = len(df)
        ndays_deg = sum(df['Avg Speed'] < 45)
        perc_deg = ndays_deg / ndays

        dict_vhtvmt = {'VMT ' + suffix: vmt,
                       'VHT ' + suffix: vht,
                       'Avg Speed ' + suffix: spd,
                       'Days with data ' + suffix: ndays,
                       'Days <45mph ' + suffix: ndays_deg,
                       '% days degraded ' + suffix: perc_deg
                       }

        return dict_vhtvmt

    def get_degradation(self):
        # Add necessary data
        self.neighbors = self.get_neighbors()
        self.raw_data = self.get_sensor_data()

        _cols = ['Flow', 'Occupancy', 'Speed']
        base_cols = ['Station', 'District', 'Freeway', 'Direction', 'Lane Type', 'Length', 'Observed', 'DATE', 'PEAK']
        bad_cols = base_cols + _cols

        # Filter peak hours & weekdays only
        f_peaks = self.raw_data['Timestamp'].str.slice(11, 19).isin(self.peak_am + self.peak_pm)
        f_dow = pd.to_datetime(self.raw_data['Timestamp']).dt.dayofweek < 5  # 0-4: Monday-Friday, 5-6: Saturday-Sunday
        self.data = self.raw_data[f_dow & f_peaks]

        # Add peak hour and day indicator for aggregation later
        self.data = self.data.assign(DATE=self.data['Timestamp'].str.slice(0, 10))
        self.data.loc[self.data['Timestamp'].str.slice(11, 19).isin(self.peak_pm), 'PEAK'] = 'PM'
        self.data.loc[self.data['Timestamp'].str.slice(11, 19).isin(self.peak_am), 'PEAK'] = 'AM'

        hov_results = []
        for bad_id in self.bad_ids['ID']:
            good_id = int(self.neighbors[self.neighbors['bad_HOV'] == bad_id]['neighbor_ML'])
            good_lane = self.neighbors[self.neighbors['bad_HOV'] == bad_id]['real_lane'].to_string(index=False).strip()
            good_cols = [good_lane + s for s in [' Flow', ' Occupancy', ' Speed']]

            if bad_id == 769745 & self.joedays:
                # Filter Sensor 769745 for direct comparison
                datepath = self.path + 'joedates_for_769745.csv'
                # datepath = 'hov_degradation/utils/joedates_for_769745.csv'
                thesensor = self.data['Station'] == 769745
                thedates = self.data['DATE'].isin(pd.read_csv(datepath, header=None)[0].tolist())
                bad_data = self.data[thedates & thesensor][bad_cols]
            else:
                # Extract associated data & rename to match
                bad_data = self.data[self.data['Station'] == bad_id][bad_cols]

            good_data = self.data[self.data['Station'] == good_id][base_cols + good_cols]
            good_data.rename(columns={good_cols[i]: _cols[i] for i in range(len(good_cols))}, inplace=True)

            # get VMT & VMT
            output_meta = {
                'Fwy': self.meta[self.meta['ID'] == good_id]['Fwy'].to_string(index=False).strip(),
                'Direction': self.meta[self.meta['ID'] == good_id]['Dir'].to_string(index=False).strip(),
                'County': self.meta[self.meta['ID'] == good_id]['County'].to_string(index=False).strip(),
                'State_PM': self.meta[self.meta['ID'] == good_id]['State_PM'].to_string(index=False).strip(),
                'Abs_PM': self.meta[self.meta['ID'] == good_id]['Abs_PM'].to_string(index=False).strip(),
                'Name': self.meta[self.meta['ID'] == good_id]['Name'].to_string(index=False).strip(),
                'Erroneous HOV ID': bad_id,
                'Correct ID from ML': good_id,
                'Correct ML lane #': good_lane,
                'Length': self.meta[self.meta['ID'] == good_id]['Length'].to_string(index=False).strip(),
            }
            output_results = {**self.get_vhtvmt(bad_data[bad_data['PEAK'] == 'AM'], suffix='(Erroneous AM)'),
                              **self.get_vhtvmt(bad_data[bad_data['PEAK'] == 'PM'], suffix='(Erroneous PM)'),
                              **self.get_vhtvmt(good_data[good_data['PEAK'] == 'AM'], suffix='(Corrected AM)'),
                              **self.get_vhtvmt(good_data[good_data['PEAK'] == 'PM'], suffix='(Corrected PM)'),
                              }
            output = {**output_meta, **output_results}

            #add results to data frame
            hov_results.append(pd.DataFrame([output]))

        df_results = pd.concat(hov_results)

        # Use output to resort the col order that got fucked up
        df_results = df_results[dict.keys(output)]

        self.results = df_results
        return df_results

def get_dates(path):
    dates = [file for file in os.listdir(path) if 'station_hour' in file]
    dates = list(map(lambda st: str.replace(st, "d07_text_station_hour_", ""), dates))
    dates = list(map(lambda st: str.replace(st, "_", "-"), dates))
    dates = list(map(lambda st: str.replace(st, ".txt.gz", ""), dates))
    dates = dates[0] + '_to_' + dates[len(dates)-1]
    return dates

def get_holidays(path):
    # Initialize the dates
    joedates = pd.read_csv(path + 'joedates_for_769745.csv',header=None)[0]
    joedates = pd.to_datetime(joedates, format='%m/%d/%Y')
    alldates = pd.date_range(joedates.min(), joedates.max())
    # Holidays
    cal = USFederalHolidayCalendar()    # cal = calendar()
    holidays = cal.holidays(start=alldates.min(), end=alldates.max())

    # Make into DataFrame
    df = pd.DataFrame({'Date': alldates})
    df['JoeDate'] = df['Date'].isin(joedates)
    df['Weekday'] = df['Date'].dt.dayofweek < 5  # 0-4: Monday-Friday, 5-6: Saturday-Sunday
    df['Holiday'] = df['Date'].isin(holidays)

    return df

if __name__ == '__main__':
    inpath = "../../experiments/raw data/D7/hourly/"
    outpath = "../../experiments/district_7/results/"
    # inpath = "experiments/raw data/D7/hourly/"
    # outpath = "experiments/district_7/results/"

    # df_dates = get_holidays(inpath)

    # Peak hours
    peak_hours = {'peak_am': ['06:00:00', '07:00:00', '08:00:00'],
                  'peak_pm': ['15:00:00', '16:00:00', '17:00:00']}

    # These are the bad IDs and suspected mislabeled lane
    df_bad = pd.DataFrame(
        {'ID': [717822, 718270, 718313, 762500, 762549, 768743, 769238, 769745, 774055],
         'issue': ['Misconfigured']*9,
         'real_lane': ['Lane 1', 'Lane 2', 'Lane 1', 'Lane 2', 'Lane 1', 'Lane 4', 'Lane 1', 'Lane 3', 'Lane 1']}
    )

    degraded = GetDegradation(inpath, df_bad, peak_hours, saved=True, joedays=False)
    results = degraded.get_degradation()

    # Saving
    dates = get_dates(inpath)
    results.to_csv(outpath + 'degradation_results_' + dates + '.csv', index=False)




