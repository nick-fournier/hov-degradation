from preprocess.data_preprocess import PreProcess
from reporting.plot import PlotMisconfigs
from reporting.img_to_doc import PlotsToDocx
from analysis.degradation import GetDegradation, reconfigs_to_json
from analysis.train import Detection

import os
import pandas as pd


def main(inpath_detection, inpath_degradation, outpath, plot_date):

    if not inpath_detection:
        inpath_detection = input("Enter the directory of input data: ")
    if not outpath:
        inpath_detection = input("Enter location for processed output directory: ")

    #### PREPROCESSING ####
    #Check if preprocessed already
    if os.listdir(outpath + "processed"):
        flist = pd.Series(os.listdir(outpath + "processed"))
        dates = [x[-28::].replace(".csv", "") for x in flist[flist.str.contains('processed')]]
        dates = pd.Series(dates).unique()[0].replace("_", " ")
        pp = ''
        while any([pp is 'n', pp is 'y']) is False:
            pp = input("Processed data already exists for " + dates +
                       ". Do you want to run pre-processing again (y/n)?:")
            if len(pp) > 1:
                pp = pp[0].lower()
            if pp is "y":
                PP = PreProcess(inpath=inpath_detection, outpath=outpath)
                PP.save()
                dates = PP.start_date + "_to_" + PP.end_date
    else:
        PP = PreProcess(inpath=inpath_detection, outpath=outpath)
        PP.save()
        dates = PP.start_date + "_to_" + PP.end_date

    fdates = dates.replace(" ", "_")

    #### TRAINING ####
    detections = Detection(inpath=inpath_detection, outpath=outpath, date_range_string=fdates)
    detections.save()

    #### PLOTS ####
    #Generate plot files
    if not plot_date:
        plot_date = ''
        while plot_date is '':
            plot_date = input("Enter date to use for plots (yyyy-mm-dd):")

    if os.listdir(outpath + 'results/misconfig_plots_' + fdates):
        pp = ''
        while any([pp is 'n', pp is 'y']) is False:
            pp = input("Plots already exists, regenerate plots? (y/n):")
            if len(pp) > 1:
                pp = pp[0].lower()
            if pp is "y":
                plots = PlotMisconfigs(inpath=inpath_detection, outpath=outpath, plot_date=plot_date, date_range_string=fdates)
                plots.save_plots()

    #Create word document out of plots for easier viewing
    if not any(pd.Series(os.listdir(outpath + 'results/')).str.contains('HOV plots')):
        pp = ''
        while any([pp is 'n', pp is 'y']) is False:
            pp = input("Docx already exists, regenerate Docx? (y/n):")
            if len(pp) > 1:
                pp = pp[0].lower()
            if pp is "y":
                document = PlotsToDocx(outpath=outpath, plot_date=plot_date, date_range_string=fdates)
                document.save()


    #### DEGRADATION ####
    if not os.path.isfile(outpath + 'fixed_sensor_labels.json'):
        # These are the bad IDs and suspected mislabeled lane
        reconfigs = {'ID': [717822, 718270, 718313, 762500, 762549, 768743, 769238, 769745, 774055],
                     'issue': ['Misconfigured']*9,
                     'real_lane': ['Lane 1', 'Lane 2', 'Lane 1', 'Lane 2',
                                   'Lane 1', 'Lane 4', 'Lane 1', 'Lane 3', 'Lane 1']}
        df_bad = pd.DataFrame(reconfigs)
        df_bad.to_csv(outpath + 'results/fixed_sensor_labels.csv', index=False)
        reconfigs_to_json(outpath + 'results/fixed_sensor_labels.json', df_bad)
    else:
        # df_bad = pd.read_json(outpath + 'results/fixed_sensor_labels.json')
        df_bad = pd.read_csv(outpath + 'results/fixed_sensor_labels.csv')

    if os.path.isfile(outpath + 'results/degradation_results_' + dates + '.csv'):
        pp = ''
        while any([pp is 'n', pp is 'y']) is False:
            pp = input("Degredation results already exist, run it again? (y/n):")
            if len(pp) > 1:
                pp = pp[0].lower()
            if pp is "y":
                degraded = GetDegradation(inpath_degradation, outpath, df_bad, saved=False)
                degraded.save(dates)
    else:
        degraded = GetDegradation(inpath_degradation, outpath, df_bad, saved=True)
        degraded.save(dates)

    print("HOV Degradation Analysis Complete")

if __name__ == "__main__":
    main(inpath_detection="../experiments/input/D7/5min/2021_03_01-07/",
         inpath_degradation="../experiments/input/D7/hourly/2020/",
         outpath="../experiments/output/",
         plot_date="2021-03-03"  # This is a wednesday
         )
