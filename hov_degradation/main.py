from preprocess.data_preprocess import PreProcess
from train.train import Detection
from reporting.plot import PlotMisconfigs
from reporting.img_to_doc import PlotsToDocx

import os
import pandas as pd

def main(inpath="../experiments/input/D7/5min/2021_03_01-07/",
         outpath="../experiments/output/"):

    if not inpath:
        inpath = input("Enter the directory of input data: ")
    if not outpath:
        outpath = input("Enter location for processed output directory: ")

    #Check if preprocessed already
    if os.listdir(outpath + "processed"):
        flist = pd.Series(os.listdir(outpath + "processed"))
        dates = [x[-28::].replace(".csv", "") for x in flist[flist.str.contains('processed')]]
        dates = pd.Series(dates).unique()[0].replace("_", " ")
        pp = ''
        while any([pp is 'n', pp is 'y']) is False:
            pp = input("Processed data already exists for " + dates +
                       ". Do you want use this data (y) or run pre-processing again (n)?:")
            if len(pp) > 1:
                pp = pp[0].lower()
            if pp is "n":
                PP = PreProcess(inpath=inpath, outpath=outpath)
                PP.save()
                dates = PP.start_date + "_to_" + PP.end_date
    else:
        PP = PreProcess(inpath=inpath, outpath=outpath)
        PP.save()
        dates = PP.start_date + "_to_" + PP.end_date

    fdates = dates.replace(" ", "_")
    plot_date = "2021-03-03"  # This is a wednesday

    #Train the data for the dates in the data
    detections = Detection(inpath=inpath, outpath=outpath, date_range_string=fdates)
    detections.save()

    #Generate plots
    if os.listdir(outpath + 'results/misconfig_plots_' + fdates):
        pp = ''
        while any([pp is 'n', pp is 'y']) is False:
            pp = input("Plots already exists, regenerate plots? (y/n):")
            if len(pp) > 1:
                pp = pp[0].lower()
            if pp is "y":
                plots = PlotMisconfigs(inpath=inpath, outpath=outpath, plot_date=plot_date, date_range_string=fdates)
                plots.save_plots()

    #Create word document out of plots for easier viewing
    if not any(pd.Series(os.listdir(outpath + 'results/')).str.contains('HOV plots')):
        pp = ''
        while any([pp is 'n', pp is 'y']) is False:
            pp = input("Docx already exists, regenerate plots? (y/n):")
            if len(pp) > 1:
                pp = pp[0].lower()
            if pp is "y":
                document = PlotsToDocx(outpath=outpath, plot_date=plot_date, date_range_string=fdates)
                document.save()

if __name__ == "__main__":
    main(inpath="experiments/input/D7/5min/2021_03_01-07/",
         outpath="experiments/output/")
