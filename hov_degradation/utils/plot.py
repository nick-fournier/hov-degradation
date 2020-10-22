"""Script for plotting the results"""
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os


def save_plots(df_data, df_meta, neighbors, misconfig_ids, path):
    # load data
    # path = "../../experiments/district_7/"
    # with open(path + "neighbors_D7.json") as f:
    #     neighbors = json.load(f)
    # df_data = pd.read_csv(path + "data/station_5min_2020-05-24.csv")
    # df_meta = pd.read_csv(path + "data/meta_2020-05-23.csv")
    #
    #
    # train_data = pd.read_csv(path + "prdictions_D7_train.csv", index_col=0)
    # test_data = pd.read_csv(path + "prdictions_D7_test.csv", index_col=0)
    # misconfig_ids = list(train_data[train_data['preds'] == 1].index) + list(test_data[test_data['preds'] == 1].index)

    for misconfig_id in misconfig_ids:
        # neighbors
        up_neighbor = neighbors[str(misconfig_id)]['up']
        down_neighbor = neighbors[str(misconfig_id)]['down']
        main_neighbor = neighbors[str(misconfig_id)]['main']

        # number of lanes in the main line
        main_num_lanes = df_meta[df_meta['ID'] == main_neighbor]['Lanes'].iloc[0]

        # data frames
        _df = df_data[df_data['Station'] == misconfig_id]
        _df_up = df_data[df_data['Station'] == up_neighbor]
        _df_down = df_data[df_data['Station'] == down_neighbor]
        _df_main = df_data[df_data['Station'] == main_neighbor]

        # create output directory
        outdir = path + "/{}".format(misconfig_id)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # plt.figure(figsize=(8, 6))
        # ax = plt.subplot(4, 1, 1)
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

        # with upstream
        plt.plot()
        plt.title("HOV: {} vs Upstream :{}".format(misconfig_id, up_neighbor))
        plt.xlabel('Time')
        plt.xticks([])
        plt.ylabel('Flow')
        plt.plot(_df['Timestamp'], _df['Flow'],
                 label='HOV')
        plt.plot(_df_up['Timestamp'], _df_up['Flow'],
                 label='Upstream')
        plt.legend()
        plt.savefig(outdir + '/upstream.png', dpi=500)
        plt.close()

        # with downstream
        plt.plot()
        plt.title("HOV: {} vs Downstream :{}".format(misconfig_id, down_neighbor))
        plt.xlabel('Time')
        plt.xticks([])
        plt.ylabel('Flow')
        plt.plot(_df['Timestamp'], _df['Flow'],
                 label='HOV')
        plt.plot(_df_down['Timestamp'], _df_down['Flow'],
                 label='Downstream')
        plt.legend()
        plt.savefig(outdir + '/downstream.png', dpi=500)
        plt.close()

        # mainline lanes
        plt.subplot(211)
        plt.title("HOV: {}".format(misconfig_id))
        plt.ylabel('Flow')
        plt.xticks([])
        plt.plot(_df['Timestamp'], _df['Flow'],
                 label='HOV')
        plt.subplot(212)
        plt.title("Mainline: ".format(main_neighbor))
        plt.xlabel('Time')
        plt.xticks([])
        for n in range(1, main_num_lanes + 1):
            plt.plot(_df_main['Timestamp'], _df_main['Lane {} Flow'.format(n)],
                     label='Lane: {}'.format(n))

        plt.legend()
        plt.savefig(outdir + '/main.png', dpi=500)
        plt.close()




