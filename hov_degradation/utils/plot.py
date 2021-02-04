"""Script for plotting the results"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from cycler import cycler
from palettable.colorbrewer.qualitative import Set1_7

class PlotMisconfigs:

    def __init__(self, path, plot_date, data_dates):
        self.path = path
        self.plot_date = plot_date
        self.data_dates = data_dates
        self.data = None
        self.meta = None
        self.neighbors = None
        self.dict_mis_ids = None
        self.df_mis_ids = None
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
            self.dict_mis_ids = json.load(f)

        # Get unique predictions
        mis_ids_unique = self.dict_mis_ids['classification'] + self.dict_mis_ids['unsupervised']
        mis_ids_unique = pd.Series(mis_ids_unique).sort_values().unique().tolist()

        # Create nice data frame
        self.df_mis_ids = pd.DataFrame()
        for id in mis_ids_unique:
            method = [m for m in ['classification', 'unsupervised'] if id in self.dict_mis_ids[m]]
            method = ' & '.join(method)
            self.df_mis_ids = self.df_mis_ids.append(pd.DataFrame({'id': [id], 'method': [method]}))

    # Blah we don't need this right now
    def get_onramps(self):
        # df_group_id = self.meta.groupby('ID').first()   # All sensors
        # df_bad_id = df_group_id[df_group_id.index.isin(self.df_mis_ids['id'])]  # Only misconfigured sensors
        #
        # # Find sensors that are between
        # ramp_id = []
        # for i, _id in enumerate(df_bad_id.index):
        #     id_up = self.neighbors[str(_id)]['up']
        #     id_down = self.neighbors[str(_id)]['down']
        #     pm_up = float(df_group_id[df_group_id.index == id_up]['Abs_PM'])
        #     pm_down = float(df_group_id[df_group_id.index == id_down]['Abs_PM'])
        #
        #     f_fwy = df_group_id['Fwy'] == df_bad_id['Fwy'].loc[_id]
        #     f_dir = df_group_id['Dir'] == df_bad_id['Dir'].loc[_id]
        #     f_between = df_group_id['Abs_PM'].between(pm_up, pm_down, inclusive=False) | \
        #                 df_group_id['Abs_PM'].between(pm_down, pm_up, inclusive=False)
        #     f_ramp = df_group_id['Type'] == 'OR'
        #
        #     ramp_id.append(df_group_id[f_fwy & f_dir & f_between & f_ramp].index[0])
        #
        #
        # df_neighbors = pd.DataFrame({'bad_HOV': df_bad_id.index.to_list(),
        #                              'neighbor_ML': ramp_id})
        # df_neighbors = df_neighbors.merge(self.misconfig_ids, left_on='bad_HOV', right_on='ID').drop(columns='ID')
        #
        # # self.neighbors = df_neighbors
        # return df_neighbors
        pass

    def strip_map(self):
        pass

    def save_plots(self):
        # Plot each prediction
        date_string = pd.to_datetime(self.plot_date).day_name()[0:3] + ' ' + self.plot_date
        colors = Set1_7.mpl_colors

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
            #          linewidth=0.5, label='Downstream: {}'.format(down_neighbor))
            # plt.plot(_df['Timestamp'], _df['Flow'],
            #          color='black', label='HOV sensor: {}'.format(mis_id))
            # plt.legend()
            # plt.savefig(outdir + '/{}_combo.png'.format(mis_id), dpi=300, bbox_inches='tight')
            # plt.close()

            ####  Longitudinal plot (up vs down) ####
            plt.figure(figsize=(5, 5))
            plt.plot(colors=colors)
            plt.rc('font', size=8)
            plt.suptitle("Longitudinal comparison of {} for {}".format(mis_id, date_string))
            plt.title("Predicted by" + this_method + " method")
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
            plt.figure(figsize=(5, 5))
            plt.plot(colors=colors)
            plt.rc('font', size=8)
            plt.suptitle("Lateral comparison of {} for {}".format(mis_id, date_string))
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
            plt.savefig(outdir + '/{}_lat.png'.format(mis_id), dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    # path = 'experiments/district_7/'
    path = '../../experiments/district_7/'
    dates = '2020-12-06_to_2020-12-12'

    plots = PlotMisconfigs(path=path, plot_date="2020-12-09", data_dates=dates)
    plots.save_plots()