"""Script for plotting the results"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from cycler import cycler
from palettable.colorbrewer.qualitative import Set1_7

def cm_to_inch(value):
    return value/2.54

class PlotMisconfigs:

    def __init__(self, path, plot_date, data_dates):
        self.path = path
        self.plot_date = plot_date
        self.data_dates = data_dates
        self.data = None
        self.meta = None
        self.neighbors = None
        self.misconfig_ids = None
        self.get_data()

    def get_data(self):
        print("Loading traffic data...")
        # Load the table data as CSV
        self.data = pd.read_csv(self.path + "data/station_5min_" + self.plot_date + ".csv")
        self.meta = pd.read_csv(self.path + "data/meta_2020-11-16.csv")

        # Load JSON files
        with open(self.path + "neighbors_D7_" + self.data_dates + ".json") as f:
            self.neighbors = json.load(f)

        with open(self.path + "results/misconfigured_ids_D7_" + self.data_dates + ".json") as f:
            self.misconfig_ids = json.load(f)

        print("Done")

    def save_plots(self):
        """
        Parameters
        ----------
        df_data :
        df_meta :
        neighbors :
        misconfig_ids :
        path :

        Returns
        -------
        """

        # Get unique predictions
        mis_ids_unique = self.misconfig_ids['classification'] + self.misconfig_ids['unsupervised']
        mis_ids_unique = pd.Series(mis_ids_unique).sort_values().unique().tolist()

        date_string = pd.to_datetime(self.plot_date).day_name() + ', ' + self.plot_date
        colors = Set1_7.mpl_colors

        # Create nice data frame
        df_mis_id = pd.DataFrame()
        for id in mis_ids_unique:
            method = [m for m in ['classification', 'unsupervised'] if id in self.misconfig_ids[m]]
            method = ' & '.join(method)
            df_mis_id = df_mis_id.append(pd.DataFrame({'id': [id], 'method': [method]}))


        for mis_id in mis_ids_unique:
            # neighbors
            up_neighbor = self.neighbors[str(mis_id)]['up']
            down_neighbor = self.neighbors[str(mis_id)]['down']
            main_neighbor = self.neighbors[str(mis_id)]['main']
            this_method = df_mis_id[df_mis_id['id'] == mis_id].method.to_string(index=False)

            # number of lanes in the main line
            main_num_lanes = self.meta[self.meta['ID'] == main_neighbor]['Lanes'].iloc[0]

            # data frames
            _df = self.data[self.data['Station'] == mis_id]
            _df_up = self.data[self.data['Station'] == up_neighbor]
            _df_down = self.data[self.data['Station'] == down_neighbor]
            _df_main = self.data[self.data['Station'] == main_neighbor]

            # create output directory
            outdir = self.path + "results/misconfig_plots_" + self.data_dates + "/{}".format(mis_id)
            if not os.path.exists(outdir):
                os.makedirs(outdir)


            plt.ioff()  # Turns of interative plot display.

            # #### Combo plot ####
            # plt.figure(figsize=(8, 4))
            # plt.plot(colors=colors)
            # plt.rc('font', size=8)
            # plt.suptitle("Comparison of HOV sensor: {} on {}".format(mis_id, date_string))
            # plt.title("Predicted by" + this_method + " method")
            # plt.xlabel('Time')
            # plt.xticks([])
            # plt.ylabel('Flow')
            # plt.ylim(0, 250)
            # for n in range(1, main_num_lanes + 1):
            #     plt.plot(_df_main['Timestamp'], _df_main['Lane {} Flow'.format(n)],
            #              linewidth=0.75, linestyle='dotted', label='Mainline lane: {}'.format(n))
            # plt.plot(_df_up['Timestamp'], _df_up['Flow'],
            #          linewidth=0.5, label='Upstream: {}'.format(up_neighbor))
            # plt.plot(_df_down['Timestamp'], _df_down['Flow'],
            #          linewidth=0.5, label='Uownstream: {}'.format(down_neighbor))
            # plt.plot(_df['Timestamp'], _df['Flow'],
            #          color='black', label='HOV sensor: {}'.format(mis_id))
            # plt.legend()
            # plt.savefig(outdir + '/{}_combo.png'.format(mis_id), dpi=500)
            # plt.close()

            ####  Longitudinal plot (up vs down) ####
            plt.figure(figsize=(8, 4))
            plt.plot(colors=colors)
            plt.rc('font', size=8)
            plt.suptitle("Longitudinal comparison of HOV sensor: {} on {}".format(mis_id, date_string))
            plt.title("Predicted by" + this_method + " method")
            plt.xlabel('Time')
            plt.xticks([])
            plt.ylabel('Flow')
            # plt.ylim(0, 250)
            plt.plot(_df_up['Timestamp'], _df_up['Flow'],
                     alpha=0.5, linewidth=0.5, label='Upstream: {}'.format(up_neighbor))
            plt.plot(_df_down['Timestamp'], _df_down['Flow'],
                     alpha=0.5, linewidth=0.5, label='Uownstream: {}'.format(down_neighbor))
            plt.plot(_df['Timestamp'], _df['Flow'],
                     color='black', label='HOV sensor: {}'.format(mis_id))
            plt.legend()
            plt.savefig(outdir + '/{}_long.png'.format(mis_id), dpi=500)
            plt.close()

            #### Lateral plot (HOV vs mainline) ####
            plt.figure(figsize=(8, 4))
            plt.plot(colors=colors)
            plt.rc('font', size=8)
            plt.suptitle("Lateral comparison of HOV sensor: {} on {}".format(mis_id, date_string))
            plt.title("Predicted by" + this_method + " method")
            plt.xlabel('Time')
            plt.xticks([])
            plt.ylabel('Flow')
            # plt.ylim(0, 250)
            for n in range(1, main_num_lanes + 1):
                plt.plot(_df_main['Timestamp'], _df_main['Lane {} Flow'.format(n)],
                         alpha=0.5, linewidth=0.75, label='Mainline lane: {}'.format(n))
            plt.plot(_df['Timestamp'], _df['Flow'],
                     color='black', label='HOV sensor: {}'.format(mis_id))
            plt.legend()
            plt.savefig(outdir + '/{}_lat.png'.format(mis_id), dpi=500)
            plt.close()

if __name__ == '__main__':
    path = 'experiments/district_7/'
    dates = '2020-12-06_to_2020-12-12'

    plots = PlotMisconfigs(path=path, plot_date="2020-12-09", data_dates=dates)
    plots.save_plots()