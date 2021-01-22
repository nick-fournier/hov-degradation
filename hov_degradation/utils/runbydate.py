import pandas as pd
import pems.download
import datetime
import os

from datetime import datetime
from pems.download import PemsDownloader as PDR

class RunByDate:
    def __init__(self, start_date, end_date, location='district_7'):
        # if not start_date:
         # start_date = datetime.date.today()
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        self.location = location
        self.path = '../../experiments/' + location

    def download(self):
        # Initialize
        pdr = PDR(username='cc_data@berkeley.edu', password='CC#dss2017')

        if not os.path.exists(self.path + '/data/'):
            os.makedirs(self.path + '/data/')

        day, df_meta = pdr.download('meta')  # , username='cc_data@berkeley.edu', password='CC#dss2017')
        metapath = self.path + '/data/meta_' + str(day) + '.csv'
        df_meta.to_csv(metapath, index=False, header=True)

        dates = pd.date_range(self.start_date, self.end_date)
        for date in dates:
            path = self.path + '/data/station_5min_' + str(date.date()) + '.csv'
            if not os.path.exists(path):
                day, df_day = pdr.download('station_5min', date=date.date())
                df_day.to_csv(path, index=False, header=True)


    def analysis(self):
        pass



if __name__ == '__main__':

    #print( os.listdir('../..') )
    #os.mkdir("../../experiments/5min/data/")
    data = RunByDate('2020-10-25', '2020-10-31').download()



    # path = "../../experiments/district_7/data/"
    # df_data = pd.read_csv(path + "station_5min_2020-05-24.csv")
    # df_meta = pd.read_csv(path + "meta_2020-05-23.csv")
    # data = PreProcess(df_data, df_meta, location='i210')
    # df_train, df_test, neighbors_i210 = data.preprocess()
    # df_train.to_csv(path[:-5] + "processed_i210_train.csv")
    # df_test.to_csv(path[:-5] + "processed_i210_test.csv")
    #
    # # District 7
    # data = PreProcess(df_data, df_meta, location='5min', split=False)
    # df_D7, _, neighbors_D7 = data.preprocess()
    # df_D7.to_csv(path[:-5] + "processed_D7.csv")
    # with open(path[:-5] + 'neighbors_D7.json', 'w') as f:
    #     json.dump(neighbors_D7, f, sort_keys=True, indent=4)