"""Script for plotting the results"""
import matplotlib.pyplot as plt
import os

def get_data():
    pass

def save_plots(df_data, df_meta, neighbors, misconfig_ids, path):
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
        plt.title("Mainline: {}".format(main_neighbor))
        plt.xlabel('Time')
        plt.xticks([])
        for n in range(1, main_num_lanes + 1):
            plt.plot(_df_main['Timestamp'], _df_main['Lane {} Flow'.format(n)],
                     label='Lane: {}'.format(n))

        plt.legend()
        plt.savefig(outdir + '/main.png', dpi=500)
        plt.close()




