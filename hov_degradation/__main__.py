from .preprocess.data_preprocess import PreProcess
from .reporting.plot import PlotMisconfigs
from .reporting.img_to_doc import PlotsToDocx
from .analysis.degradation import GetDegradation, reconfigs_to_json
from .analysis.train import Detection

import numpy.random.common
import numpy.random.bounded_integers
import numpy.random.entropy
import os
import pandas as pd


def main(inpath_detection=None,
         inpath_degradation=None,
         outpath=None,
         plot_date=None
         ):

    print("Welcome to erroneous HOV detection and degradatation analysis!")

    degradation = None
    while not inpath_detection:
        degradation = input("Do you want to run erroneous detection (1) or degradation analysis (2)?\r"
                          "(Enter '1' or '2' for detection or degradation\r"
                          "NOTE: You must provide a corrected HOV sensor labels before running degradation analysis.")
        if degradation != 1 or degradation != 2:
            degradation = None


    #### PREPROCESSING ####
    if degradation == 1:

        while not inpath_detection:
            inpath_detection = input("Enter the directory of 5-min traffic count input data: ")
            if not os.path.isdir(inpath_detection) or len(os.listdir(inpath_detection)) == 0:
                print("Invalid input path")
                inpath_detection = None

        while not outpath:
            outpath = input("Enter location for processed output directory: ")

        while not plot_date:
            plot_date = input("Enter date to use for output plots (yyyy-mm-dd): ")

        #Check if preprocessed already
        print("Running Preprocessing...")
        if os.listdir(outpath + "processed"):
            flist = pd.Series(os.listdir(outpath + "processed"))
            dates = [x[-28::].replace(".csv", "") for x in flist[flist.str.contains('processed')]]
            dates = pd.Series(dates).unique()[0].replace("_", " ")
            pp = ''
            while any([pp is 'n', pp is 'y']) is False:
                pp = input("Processed data already exists for " + dates +
                           ". Do you want to run pre-processing again (y/n)?: ")
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

        #### ANALYSIS ####
        print("Running analysis...")
        detections = Detection(inpath=inpath_detection, outpath=outpath, date_range_string=fdates)
        detections.save()

        #### PLOTS ####
        print("Plotting results...")
        #Generate plot files
        if os.listdir(outpath + 'results/plots_misconfigs_' + fdates):
            pp = ''
            while any([pp is 'n', pp is 'y']) is False:
                pp = input("Plots already exists, regenerate plots? (y/n): ")
                if len(pp) > 1:
                    pp = pp[0].lower()
                if pp is "y":
                    plots = PlotMisconfigs(inpath=inpath_detection, outpath=outpath, plot_date=plot_date, date_range_string=fdates)
                    plots.save_plots()

        print("Printing to word doc...")
        #Create word document out of plots for easier viewing
        if not any(pd.Series(os.listdir(outpath + 'results/')).str.contains('docx')):
            pp = ''
            while any([pp is 'n', pp is 'y']) is False:
                pp = input("Docx already exists, regenerate Docx? (y/n): ")
                if len(pp) > 1:
                    pp = pp[0].lower()
                if pp is "y":
                    document = PlotsToDocx(outpath=outpath, plot_date=plot_date, date_range_string=fdates)
                    document.save()

    else:
        #### DEGRADATION ####
        while not inpath_degradation:
            inpath_degradation = input("Enter the directory of hourly traffic count input data for degradation analysis: ")
            if not os.path.isdir(inpath_detection) or len(os.listdir(inpath_detection)) == 0:
                print("Invalid input path")
                inpath_detection = None

        while not outpath:
            outpath = input("Enter location for processed output directory: ")

        print("Running degradation analysis...")

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

        if any(pd.Series(os.listdir(outpath + 'results')).str.contains('degradation_results')):
            pp = ''
            while any([pp is 'n', pp is 'y']) is False:
                pp = input("Degredation results already exist, run it again? (y/n): ")
                if len(pp) > 1:
                    pp = pp[0].lower()
                if pp is "y":
                    degraded = GetDegradation(inpath_degradation, outpath, df_bad, saved=False)
                    degraded.save()
        else:
            degraded = GetDegradation(inpath_degradation, outpath, df_bad, saved=True)
            degraded.save()

    print("Analysis Complete")
    exit = 'n'
    while exit == 'n':
        exit = input("Exit? (y/n): ")


if __name__ == "__main__":
    main()
    # main(inpath_detection="../experiments/input/D7/5min/2021_03_01-07/",
    #      inpath_degradation="../experiments/input/D7/hourly/2020/",
    #      outpath="../experiments/output/",
    #      plot_date="2021-03-03"  # This is a wednesday
    #      )
