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
import warnings

def get_fixed(path):
    if not os.path.isfile(path + 'fixed_sensor_labels.json'):
        # These are the bad IDs and suspected mislabeled lane
        reconfigs = {'ID': [717822, 718270, 718313, 762500, 762549, 768743, 769238, 769745, 774055],
                     'issue': ['Misconfigured'] * 9,
                     'real_lane': ['Lane 1', 'Lane 2', 'Lane 1', 'Lane 2',
                                   'Lane 1', 'Lane 4', 'Lane 1', 'Lane 3', 'Lane 1']}
        df_bad = pd.DataFrame(reconfigs)
        df_bad.to_csv(path + 'fixed_sensor_labels.csv', index=False)
        reconfigs_to_json(path + 'fixed_sensor_labels.json', df_bad)
    else:
        # df_bad = pd.read_json(outpath + 'fixed_sensor_labels.json')
        df_bad = pd.read_csv(path + 'fixed_sensor_labels.csv')

    return df_bad

def check_path(path):
    if path[-1] is not "/":
        return path + "/"
    else:
        return path

class main:
    def __init__(self,
                 detection_data_path=None,
                 degradation_data_path=None,
                 output_path=None,
                 plotting_date=None
                 ):
        print("Welcome to erroneous HOV detection and degradation analysis!")
        exit = None
        while exit != 'y':
            degradation = None
            while not degradation:
                degradation = input("Do you want to run erroneous detection (1) or degradation analysis (2)?: ")
                if degradation != '1' and degradation != '2':
                    degradation = None

            warnings.filterwarnings('ignore')

            #### PREPROCESSING ####
            if degradation == '1':
                while not detection_data_path:
                    detection_data_path = input("Enter the directory of 5-min traffic count input data: ")
                self.inpath_detection = check_path(detection_data_path)

                while not output_path:
                    output_path = input("Enter location for processed output directory: ")
                self.outpath = check_path(output_path)

                while not plotting_date:
                    plotting_date = input("Enter date to use for output plots (yyyy-mm-dd): ")
                    if not any([plotting_date.replace('-', '_') in x for x in os.listdir(self.inpath_detection)]):
                        print("not a valid date in data range")
                        plotting_date = None
                self.plot_date = plotting_date

                #Check if preprocessed already
                self.subout = list(filter(None, self.inpath_detection.split('/')))[-1]
                self.run_preprocessing()
                self.datestring = [x for x in os.listdir(self.outpath + self.subout + "/processed/") if 'neighbors' in x][0][-29:-5]

                #### ANALYSIS ####
                self.run_analysis()

                #### PLOTS ####
                self.run_plotting("Plots already exists, regenerate plots? (y/n): ")

            else:
                #### DEGRADATION ####
                while not degradation_data_path:
                    degradation_data_path = input("Enter the directory of hourly traffic count input data for degradation analysis: ")
                self.inpath_degradation = check_path(degradation_data_path)

                while not output_path:
                    output_path = input("Enter location for processed output directory: ")
                self.outpath = check_path(output_path)

                # while not plotting_date:
                #     plotting_date = input("Enter date to use for output plots (yyyy-mm-dd): ")
                # self.plot_date = plotting_date

                self.subout = list(filter(None, self.inpath_degradation.split('/')))[-1]
                self.run_degradation()

                # #### REPLOTS ####
                # self.run_plotting("Do you want to update the plots with degradation results? (y/n): ")

            while exit != "y" and exit != "n":
                exit = input("Analysis Complete, exit? (y/n): ")

    def run_preprocessing(self):
        the_path = self.outpath + self.subout + '/processed'
        if os.path.isdir(the_path) and len(os.listdir(the_path)) > 0:
            pp = ''
            # Asks if you want to run it again
            while any([pp is 'n', pp is 'y']) is False:
                pp = input("Processed data already exists for " + self.subout +
                           ". Do you want to run pre-processing (y/n)?: ")
                if len(pp) > 1:
                    pp = pp[0].lower()
                if pp is "y":
                    PreProcess(inpath=self.inpath_detection, outpath=self.outpath + self.subout).save()
        # Runs it if it doesn't exist
        else:
            PreProcess(inpath=self.inpath_detection, outpath=self.outpath + self.subout).save()

    def run_analysis(self):
        the_path = self.outpath + self.subout + '/plots_misconfigs_' + self.plot_date
        if os.path.isdir(the_path) and len(os.listdir(the_path)) > 0:
            pp = ''
            # Asks if you want to run it again
            while any([pp is 'n', pp is 'y']) is False:
                pp = input("Analysis results already exists for " + self.subout +
                           ". Do you want to run again (y/n)?: ")
                if len(pp) > 1:
                    pp = pp[0].lower()
                if pp is "y":
                    Detection(inpath=self.inpath_detection,
                              outpath=self.outpath + self.subout,
                              date_range_string=self.datestring).save()
        # Runs it if it doesn't exist
        else:
            Detection(inpath=self.inpath_detection,
                      outpath=self.outpath + self.subout,
                      date_range_string=self.datestring).save()

    def run_plotting(self, text):
        # Generate plot files
        the_path = self.outpath + self.subout + '/plots_misconfigs_' + self.plot_date
        if os.path.isdir(the_path) and len(os.listdir(the_path)) > 0:
            pp = ''
            # Asks if you want to run it again
            while any([pp is 'n', pp is 'y']) is False:
                pp = input(text)
                if len(pp) > 1:
                    pp = pp[0].lower()
                if pp is "y":
                    PlotMisconfigs(inpath=self.inpath_detection,
                                   outpath=self.outpath + self.subout,
                                   plot_date=self.plot_date,
                                   date_range_string=self.datestring)
                    PlotsToDocx(outpath=self.outpath + self.subout,
                                plot_date=self.plot_date,
                                date_range_string=self.datestring
                                ).save()

        # Runs it if it doesn't exist
        else:
            PlotMisconfigs(inpath=self.inpath_detection,
                           outpath=self.outpath + self.subout,
                           plot_date=self.plot_date,
                           date_range_string=self.datestring)

            PlotsToDocx(outpath=self.outpath + self.subout,
                        plot_date=self.plot_date,
                        date_range_string=self.datestring
                        ).save()

    def run_degradation(self):
        df_bad = get_fixed(self.outpath)
        the_path = self.outpath + self.subout + '/degradation/'
        if os.path.isdir(the_path) and len(os.listdir(the_path)) > 0:
            pp = ''
            # Asks if you want to run it again
            while any([pp is 'n', pp is 'y']) is False:
                pp = input("Degredation analysis already exist, run it again? (y/n): ")
                if len(pp) > 1:
                    pp = pp[0].lower()
                if pp is "y":
                    GetDegradation(inpath=self.inpath_degradation,
                                   outpath=self.outpath + self.subout,
                                   bad_sensors=df_bad,
                                   saved=False).save()
        # Runs it if it doesn't exist
        else:
            GetDegradation(inpath=self.inpath_degradation,
                           outpath=self.outpath + self.subout,
                           bad_sensors=df_bad,
                           saved=False).save()
