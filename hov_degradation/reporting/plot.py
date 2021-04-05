"""Script for plotting the results"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from cycler import cycler
# from palettable.colorbrewer.qualitative import Set1_7

class PlotMisconfigs:

    def __init__(self, inpath, outpath, plot_date, date_range_string):
        # Amend string
        if outpath[-1] is not "/":
            self.inpath = inpath + "/"
        else:
            self.inpath = inpath

        # Amend string
        if outpath[-1] is not "/":
            self.outpath = outpath + "/"
        else:
            self.outpath = outpath

        self.plot_date = plot_date
        self.data_dates = date_range_string
        self.data = None
        self.meta = None
        self.neighbors = None
        self.dict_mis_ids = None
        self.df_mis_ids = None
        self.reconfig_lanes = None
        self.get_data()

    def get_data(self):
        print("Loading traffic data...")
        # Checks whether it's running or in debug
        if os.path.isfile('static/5min_headers.csv'):
            headers = pd.read_csv('static/5min_headers.csv', index_col=0, header=0).columns
        else:
            headers = pd.read_csv('hov_degradation/static/5min_headers.csv', index_col=0, header=0).columns

        #Load 5min data for one day specified
        self.flist = pd.Series(os.listdir(self.inpath))
        filt = self.flist.str.contains("station_5min_" + self.plot_date.replace("-", "_") + ".txt.gz")
        f = self.flist[filt].to_list()[0]
        self.data = pd.read_csv(self.inpath + f)
        # self.data = pd.read_csv(self.inpath + "d07_text_station_5min_" + self.plot_date.replace("-", "_") + ".txt.gz")
        self.data.columns = headers

        # Meta data
        self.flist = pd.Series(os.listdir(self.inpath))
        f = self.flist[self.flist.str.contains("meta")][0]
        self.meta = pd.read_csv(self.inpath + f, sep="\t")

        # Load JSON files
        with open(self.outpath + "processed/processed_neighbors_D7_" + self.data_dates + ".json") as f:
            self.neighbors = json.load(f)

        with open(self.outpath + "results/analysis_misconfigured_ids_D7_" + self.data_dates + ".json") as f:
            self.dict_mis_ids = json.load(f)

        with open(self.outpath + "results/fixed_sensor_labels.json") as f:
            self.reconfig_lanes = json.load(f)

        # Get unique predictions
        mis_ids_unique = self.dict_mis_ids['classification'] + self.dict_mis_ids['unsupervised']
        mis_ids_unique = pd.Series(mis_ids_unique).sort_values().unique().tolist()

        # Create nice data frame
        self.df_mis_ids = pd.DataFrame()
        for id in mis_ids_unique:
            method = [m for m in ['classification', 'unsupervised'] if id in self.dict_mis_ids[m]]
            method = ' & '.join(method)
            self.df_mis_ids = self.df_mis_ids.append(pd.DataFrame({'id': [id], 'method': [method]}))

    def save_plots(self):
        # Plot each prediction
        # colors = Set1_7.mpl_colors
        colors = [(0.8941176470588236, 0.10196078431372549, 0.10980392156862745),
                  (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
                  (0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
                  (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
                  (1.0, 0.4980392156862745, 0.0),
                  (1.0, 1.0, 0.2),
                  (0.6509803921568628, 0.33725490196078434, 0.1568627450980392)]

        for c, mis_id in enumerate(list(self.df_mis_ids['id'])):
            count = str(c + 1) + '/' + str(len(self.df_mis_ids))
            print('Plotting misconfigured VDS ' + str(mis_id) + ' plot ' + count)
            # neighbors
            up_neighbor = self.neighbors[str(mis_id)]['up']
            down_neighbor = self.neighbors[str(mis_id)]['down']
            main_neighbor = self.neighbors[str(mis_id)]['main']
            this_method = self.df_mis_ids[self.df_mis_ids['id'] == mis_id].method.to_string(index=False)

            # number of lanes in the main line
            main_num_lanes = self.meta[self.meta['ID'] == main_neighbor]['Lanes'].iloc[0]

            # data frames
            _df = self.data[self.data['Station'] == mis_id]
            _df_up = self.data[self.data['Station'] == up_neighbor]
            _df_down = self.data[self.data['Station'] == down_neighbor]
            _df_main = self.data[self.data['Station'] == main_neighbor]

            # create output directory
            outdir = self.outpath + "results/plots_misconfigs_" + self.data_dates + "/{}".format(mis_id)
            if not os.path.exists(outdir):
                os.makedirs(outdir)


            plt.ioff()  # Turns of interative plot display.

            ####  Longitudinal plot (up vs down) ####
            plt.figure(figsize=(5, 4))
            plt.plot(colors=colors)
            plt.rc('font', size=8)
            plt.title("Longitudinal comparison of VDS {}".format(mis_id))
            plt.xlabel('Time')
            plt.xticks([])
            plt.ylabel('Flow')
            # plt.ylim(0, 250)
            plt.plot(_df_up['Timestamp'], _df_up['Flow'],
                     alpha=0.5, linewidth=0.5, label='Upstream: {}'.format(up_neighbor))
            plt.plot(_df_down['Timestamp'], _df_down['Flow'],
                     alpha=0.5, linewidth=0.5, label='Downstream: {}'.format(down_neighbor))
            plt.plot(_df['Timestamp'], _df['Flow'],
                     color='black', label='HOV sensor: {}'.format(mis_id))
            plt.legend()
            plt.savefig(outdir + '/{}_long.png'.format(mis_id), dpi=300, bbox_inches='tight')
            plt.close()

            #### Lateral plot (HOV vs mainline) ####
            plt.figure(figsize=(5, 4))
            plt.plot(colors=colors)
            plt.rc('font', size=8)
            plt.title("Lateral comparison of VDS {}".format(mis_id))
            plt.xlabel('Time')
            plt.xticks([])
            plt.ylabel('Flow')
            # plt.ylim(0, 250)
            for n in range(1, main_num_lanes + 1):
                plt.plot(_df_main['Timestamp'], _df_main['Lane {} Flow'.format(n)],
                         alpha=0.5, linewidth=0.75, label='Mainline lane {} sensor: {}'.format(n, main_neighbor))
            plt.plot(_df['Timestamp'], _df['Flow'],
                     color='black', label='HOV sensor: {}'.format(mis_id))
            plt.legend()
            plt.savefig(outdir + '/{}_lat.png'.format(mis_id), dpi=300, bbox_inches='tight')
            plt.close()

            #### The corrected plot ####
            if str(mis_id) in list(self.reconfig_lanes.keys()):

                n = self.reconfig_lanes[str(mis_id)]['lane_num']

                plt.figure(figsize=(8, 3))
                plt.plot(colors=colors)
                plt.rc('font', size=8)
                plt.title("Corrected lane configuration of VDS {}".format(mis_id))
                plt.xlabel('Time')
                plt.xticks([])
                plt.ylabel('Flow')
                # plt.ylim(0, 250)
                # Up/Down stream lanes
                plt.plot(_df_up['Timestamp'], _df_up['Flow'],
                         alpha=0.5, linewidth=0.5, label='Upstream: {}'.format(up_neighbor))
                plt.plot(_df_down['Timestamp'], _df_down['Flow'],
                         alpha=0.5, linewidth=0.5, label='Downstream: {}'.format(down_neighbor))
                # Misconfigured lane
                # plt.plot(_df['Timestamp'], _df['Flow'],
                #          linewidth=0.75, linestyle='dotted',
                #          color='red', label='Mislabeled HOV sensor: {}'.format(mis_id))
                # Corrected lane
                plt.plot(_df_main['Timestamp'], _df_main['Lane {} Flow'.format(n)],
                         color='green', linewidth=0.75,
                         label='Correct HOV sensor: {}, mislabeled mainline lane {}'.format(main_neighbor, n))
                plt.legend()
                plt.savefig(outdir + '/{}_fix.png'.format(mis_id), dpi=300, bbox_inches='tight')
                plt.close()