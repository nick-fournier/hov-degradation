import os
import pandas as pd
import json

from pandas.tseries.holiday import USFederalHolidayCalendar


headers = ['Timestamp', 'Station', 'District', 'Freeway', 'Direction', 'Lane Type', 'Station Length',
           'Samples', 'Observed', 'Flow', 'Occupancy', 'Speed',
           'Delay_Vt=35', 'Delay_Vt=40', 'Delay_Vt=45', 'Delay_Vt=50', 'Delay_Vt=55', 'Delay_Vt=60',
           'Lane 1 Flow', 'Lane 1 Occupancy', 'Lane 1 Speed',
           'Lane 2 Flow', 'Lane 2 Occupancy', 'Lane 2 Speed',
           'Lane 3 Flow', 'Lane 3 Occupancy', 'Lane 3 Speed',
           'Lane 4 Flow', 'Lane 4 Occupancy', 'Lane 4 Speed',
           'Lane 5 Flow', 'Lane 5 Occupancy', 'Lane 5 Speed',
           'Lane 6 Flow', 'Lane 6 Occupancy', 'Lane 6 Speed',
           'Lane 7 Flow', 'Lane 7 Occupancy', 'Lane 7 Speed',
           'Lane 8 Flow', 'Lane 8 Occupancy', 'Lane 8 Speed']

def check_path(path):
    if path[-1] is not "/":
        return path + "/"
    else:
        return path

def get_dates(path):
    dates = [file for file in os.listdir(path) if 'station_hour' in file]
    dates = list(map(lambda st: str.replace(st, "text_station_hour_", ""), dates))
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

def reconfigs_to_json(path, df):
    out = {}
    for id in df['ID']:
        df_row = df[df['ID'] == id]
        out[int(id)] = {'issue': df_row['issue'].to_string(index=False).strip(),
                   'real_lane': df_row['real_lane'].to_string(index=False).strip(),
                   'lane_num': int(df_row['real_lane'].to_string(index=False).strip()[5])}

    with open(path, 'w') as f:
        json.dump(out, f, sort_keys=True, indent=4)

def check_before_reading(path):
    with open(path) as f:
        first_line = f.readline()

    if "," in first_line:
        return pd.read_csv(path)

    if "\t" in first_line:
        return pd.read_csv(path, sep="\t")

class GetDegradation:
    def __init__(self, inpath, outpath, bad_sensors=None, saved=False):

        # Default peak hours
        peak_hours = {'peak_am': ['06:00:00', '07:00:00', '08:00:00'],
                      'peak_pm': ['15:00:00', '16:00:00', '17:00:00']}

        self.inpath = check_path(inpath)
        self.outpath = check_path(outpath)

        # Meta data
        self.flist = pd.Series(os.listdir(self.inpath))
        file = self.flist[self.flist.str.contains("meta")][0]
        self.meta = check_before_reading(self.inpath + file)

        self.district = str(self.meta.loc[0, 'District'])

        self.bad_ids = bad_sensors
        self.peak_am = peak_hours['peak_am']
        self.peak_pm = peak_hours['peak_pm']
        self.saved = saved

        self.data = self.filter_sensor_data()

        if isinstance(bad_sensors, pd.DataFrame):
            self.neighbors = self.get_neighbors()
            self.get_fixed_magnitude()
        else:
            self.get_all_degradation()

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

    def filter_sensor_data(self):
        prefiltered_path = self.outpath + 'degradation/degradation_sensors_hourly_D' + self.district + '_lastsave.csv'
        if not os.path.isdir(self.outpath + 'degradation/'):
             os.makedirs(self.outpath + 'degradation/')

        if not self.saved and os.path.isfile(prefiltered_path):
            pp = ''
            while any([pp is 'n', pp is 'y']) is False:
                pp = input("Preprocess raw hourly sensor data or use last saved? (y/n):")
                if len(pp) > 1:
                    pp = pp[0].lower()
                if pp is "n":
                    self.saved = True

        if self.saved and os.path.isfile(prefiltered_path):
            print('Loading last saved pre-processed hourly sensor data...')
            df_allsensors = pd.read_csv(prefiltered_path)
        else:
            # Headers, file list, and ID list
            # Checks whether it's running or in debug
            # if os.path.isfile('hourly_headers.csv'):
            #     headers = pd.read_csv('hourly_headers.csv', index_col=0, header=0).columns.tolist()
            # elif os.path.isfile('static/hourly_headers.csv'):
            #     headers = pd.read_csv('static/hourly_headers.csv', index_col=0, header=0).columns.tolist()
            # else:
            #     headers = pd.read_csv('hov_degradation/static/hourly_headers.csv', index_col=0, header=0).columns.tolist()

            data = []
            gzlist = pd.Series(os.listdir(self.inpath))
            gzlist = gzlist[gzlist.str.contains("txt.gz")]

            # Mainline and HOV sensor IDs
            id_list = self.meta[self.meta.Type.isin(['HV', 'ML'])].ID

            # Read file then filter, extract, and aggregate
            # Filtering is done on per-file basis as they are read in to minimize memory usage
            print('Loading raw hourly data...')
            for gzf in gzlist:
                df_hourly = pd.read_csv(self.inpath + gzf, names=headers)

                # Filter only mainline and HOV sensor IDs and if flow > 0
                df_hourly = df_hourly[df_hourly.Station.isin(id_list)]
                df_hourly = df_hourly[df_hourly.Flow > 0]

                # Get which days are weekdays in current dates
                # 0-4: Monday-Friday, 5-6: Saturday-Sunday
                dow = pd.date_range(df_hourly.Timestamp.min(), df_hourly.Timestamp.max())
                dow = dow[dow.dayofweek < 5].strftime("%m/%d/%Y").tolist()

                # Filter peak hours & weekdays
                f_peaks = df_hourly['Timestamp'].str.slice(11, 19).isin(self.peak_am + self.peak_pm)
                f_dow = df_hourly['Timestamp'].str.slice(0, 10).isin(dow)
                peakhour_data = df_hourly[f_peaks & f_dow]

                # Add peak hour and day indicator for aggregation later
                peakhour_data = peakhour_data.assign(DATE=peakhour_data['Timestamp'].str.slice(0, 10))
                peakhour_data.loc[peakhour_data['Timestamp'].str.slice(11, 19).isin(self.peak_pm), 'PEAK'] = 'PM'
                peakhour_data.loc[peakhour_data['Timestamp'].str.slice(11, 19).isin(self.peak_am), 'PEAK'] = 'AM'

                # Aggregate & remove <100 observable
                observable = peakhour_data.groupby(['Station', 'DATE', 'PEAK']).agg({'Observed': 'mean'}).reset_index()
                observable = observable[observable['Observed'] == 100]

                # Set up indices and filter matching
                keys = ['Station', 'DATE', 'PEAK']
                i1 = peakhour_data.set_index(keys).index
                i2 = observable.set_index(keys).index
                peakhour_data = peakhour_data[i1.isin(i2)]

                #Add to output list
                data.append(peakhour_data)
                print("Done loading " + gzf)

            # Bind together &  Rename Station length to get rid of space
            df_allsensors = pd.concat(data)
            df_allsensors.rename(columns={"Station Length": "Length"}, inplace=True)

            #Save
            print("Saving filtered data...")
            df_allsensors.to_csv(prefiltered_path)

        # Get start/end dates
        the_dates = pd.to_datetime([df_allsensors.Timestamp.max(), df_allsensors.Timestamp.min()])
        # the_dates = pd.to_datetime(df_allsensors.Timestamp, format='%m/%d/%Y %H:%M:%S')
        self.date_string = the_dates.min().strftime('%Y-%m') + "_to_" + the_dates.max().strftime('%Y-%m')

        return df_allsensors

    def filter_sensor_data_old(self):
        prefiltered_path = self.outpath + 'degradation/degradation_sensors_hourly_D' + self.district + '_lastsave.csv'
        if not os.path.isdir(self.outpath + 'degradation/'):
             os.makedirs(self.outpath + 'degradation/')

        if not self.saved and os.path.isfile(prefiltered_path):
            pp = ''
            while any([pp is 'n', pp is 'y']) is False:
                pp = input("Extracted hourly sensor data exists, run it again? (y/n):")
                if len(pp) > 1:
                    pp = pp[0].lower()
                if pp is "n":
                    self.saved = True

        if self.saved and os.path.isfile(prefiltered_path):
                df_sensors = pd.read_csv(prefiltered_path)
        else:
            # Headers, file list, and ID list
            # Checks whether it's running or in debug
            if os.path.isfile('static/5min_headers.csv'):
                headers = pd.read_csv('static/hourly_headers.csv', index_col=0, header=0)
            else:
                headers = pd.read_csv('hov_degradation/static/hourly_headers.csv', index_col=0, header=0)

            id_list = self.neighbors['bad_HOV'].to_list() + self.neighbors['neighbor_ML'].to_list()
            misconfigs = []
            gzlist = pd.Series(os.listdir(self.inpath))
            gzlist = gzlist[gzlist.str.contains("txt.gz")]

            # Read file, Filter for misconfig'd sensors
            for gzf in gzlist:
                df_hourly = pd.read_csv(self.inpath + gzf, header=None, names=headers)
                df_hourly = df_hourly[df_hourly['Station'].isin(id_list)]
                misconfigs.append(df_hourly)
                print("Done loading " + gzf)

            # Bind together &  Rename Station length to get rid of space
            df_sensors = pd.concat(misconfigs)
            df_sensors.rename(columns={"Station Length": "Length"}, inplace=True)

            #Save
            df_sensors.to_csv(prefiltered_path)

        # Get start/end dates
        the_dates = pd.to_datetime(df_sensors.Timestamp, format='%m/%d/%Y %H:%M:%S')
        self.date_string = the_dates.min().strftime('%Y-%m') + "_to_" + the_dates.max().strftime('%Y-%m')

        return df_sensors

    def get_vhtvmtdeg(self, df, suffix):
        # Calculate VMT and VHT
        df = df.assign(VMT=df.eval('Flow * Length'))
        df = df.assign(VHT=df.eval('(Flow * Length) / Speed'))

        # Aggregate
        grp_cols = df.columns[~df.columns.isin(['VMT', 'VHT', 'Flow', 'Occupancy', 'Speed'])].to_list()
        df = df.groupby(grp_cols).agg({'VMT': 'sum', 'VHT': 'sum'}).reset_index()

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

    def get_all_degradation(self):
        _cols = ['Flow', 'Occupancy', 'Speed']
        base_cols = ['Station', 'District', 'Freeway', 'Direction', 'Lane Type', 'Length', 'Observed', 'DATE', 'PEAK']
        hov_results = []

        # HOV IDs from meta data
        hov_ids = self.meta[self.meta.Type.isin(['HV'])].ID
        # HOV IDs in filtered data
        hov_ids = self.data[self.data.Station.isin(hov_ids)].Station.unique().tolist()

        print('Calculating degradation for all HOV sensors...')
        for _id in hov_ids:
            _data = self.data[self.data['Station'] == _id][base_cols + _cols]
            # get VMT & VMT
            output_meta = {
                'Fwy': self.meta[self.meta['ID'] == _id]['Fwy'].to_string(index=False).strip(),
                'Direction': self.meta[self.meta['ID'] == _id]['Dir'].to_string(index=False).strip(),
                'County': self.meta[self.meta['ID'] == _id]['County'].to_string(index=False).strip(),
                'State_PM': self.meta[self.meta['ID'] == _id]['State_PM'].to_string(index=False).strip(),
                'Abs_PM': self.meta[self.meta['ID'] == _id]['Abs_PM'].to_string(index=False).strip(),
                'Name': self.meta[self.meta['ID'] == _id]['Name'].to_string(index=False).strip(),
                'HOV Sensor ID': _id,
                'Length': self.meta[self.meta['ID'] == _id]['Length'].to_string(index=False).strip(),
            }
            output_results = {
                              **self.get_vhtvmtdeg(df=_data[_data['PEAK'] == 'AM'], suffix='(AM)'),
                              **self.get_vhtvmtdeg(df=_data[_data['PEAK'] == 'PM'], suffix='(PM)'),
                              }
            output = {**output_meta, **output_results}

            # add analysis to data frame
            hov_results.append(pd.DataFrame([output]))

        df_results = pd.concat(hov_results)

        # Use output to re-sort the col order
        df_results = df_results[dict.keys(output)]

        # Saving output
        df_results.to_csv(
            self.outpath + 'degradation/all_degradation_results_D' + self.district + "_" + self.date_string + '.csv',
            index=False)

    def get_fixed_degradation(self):
        _cols = ['Flow', 'Occupancy', 'Speed']
        base_cols = ['Station', 'District', 'Freeway', 'Direction', 'Lane Type', 'Length', 'Observed', 'DATE', 'PEAK']
        bad_cols = base_cols + _cols

        id_list = self.neighbors['bad_HOV'].to_list() + self.neighbors['neighbor_ML'].to_list()
        filtered_data = self.data[self.data['Station'].isin(id_list)]

        hov_results = []
        for bad_id in self.bad_ids['ID']:
            good_id = int(self.neighbors[self.neighbors['bad_HOV'] == bad_id]['neighbor_ML'])
            good_lane = self.neighbors[self.neighbors['bad_HOV'] == bad_id]['real_lane'].to_string(index=False).strip()
            good_cols = [good_lane + s for s in [' Flow', ' Occupancy', ' Speed']]

            bad_data = filtered_data[filtered_data['Station'] == bad_id][bad_cols]

            good_data = filtered_data[filtered_data['Station'] == good_id][base_cols + good_cols]
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
            output_results = {**self.get_vhtvmtdeg(bad_data[bad_data['PEAK'] == 'AM'], suffix='(Erroneous AM)'),
                              **self.get_vhtvmtdeg(bad_data[bad_data['PEAK'] == 'PM'], suffix='(Erroneous PM)'),
                              **self.get_vhtvmtdeg(good_data[good_data['PEAK'] == 'AM'], suffix='(Corrected AM)'),
                              **self.get_vhtvmtdeg(good_data[good_data['PEAK'] == 'PM'], suffix='(Corrected PM)'),
                              }
            output = {**output_meta, **output_results}

            #add analysis to data frame
            hov_results.append(pd.DataFrame([output]))

        df_results = pd.concat(hov_results)

        # Use output to resort the col order that got fucked up
        df_results = df_results[dict.keys(output)]

        # Saving output
        df_results.to_csv(
            self.outpath + 'degradation/fixed_degradation_results_D' + self.district + "_" + self.date_string + '.csv',
            index=False)

# if __name__ == "__main__":
#     self = GetDegradation(inpath="experiments/input/D7/hourly/2020/",
#                           outpath="experiments/output/2020/",
#                           bad_sensors=pd.read_csv('experiments/output/fixed_sensors.csv')
#     )
#
#     self.get_fixed_degradation()
#     self.get_all_degradation()


