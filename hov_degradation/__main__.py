"""
Copyright ©2021. The Regents of the University of California (Regents). All Rights Reserved. Permission to use, copy, modify, and distribute this software and its documentation for educational, research, and not-for-profit purposes, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright notice, this paragraph and the following two paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""

from .preprocess.data_preprocess import PreProcess
from .reporting.plot import PlotMisconfigs
from .reporting.img_to_doc import PlotsToDocx
from .analysis.degradation import GetDegradation, reconfigs_to_json
from .analysis.train import Detection

import os
import pandas as pd
import warnings
import sys

def get_fixed(path):
    if not os.path.isfile(path + 'fixed_sensors.json'):
        # These are the bad IDs and suspected mislabeled lane
        reconfigs = {'ID': [717822, 718270, 718313, 762500, 762549, 768743, 769238, 769745, 774055],
                     'issue': ['Misconfigured'] * 9,
                     'real_lane': ['Lane 1', 'Lane 2', 'Lane 1', 'Lane 2',
                                   'Lane 1', 'Lane 4', 'Lane 1', 'Lane 3', 'Lane 1']}
        df_bad = pd.DataFrame(reconfigs)
        df_bad.to_csv(path + 'fixed_sensors.csv', index=False)
        reconfigs_to_json(path + 'fixed_sensors.json', df_bad)
    else:
        # df_bad = pd.read_json(outpath + 'fixed_sensors.json')
        df_bad = pd.read_csv(path + 'fixed_sensors.csv')

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
        self.plot_date = plotting_date
        self.inpath_degradation = check_path(degradation_data_path)
        self.inpath_detection = detection_data_path
        self.outpath = check_path(output_path)

        print("Welcome to erroneous HOV detection and degradation analysis!")
        exit = None
        while exit != 'y':
            degradation = None
            while not degradation:
                degradation = input("Do you want to run:\n"
                                    "(1) Erroneous HOV sensor detection,\n"
                                    "(2) HOV degradation analysis?,\n"
                                    "(3) Magnitude of erroneous sensor degradation analysis?, or\n"
                                    "(E) Exit?: ").upper()
                if degradation != '1' and degradation != '2' and degradation != '3' and degradation != 'E':
                    degradation = None

            if degradation == 'E':
                sys.exit()

            warnings.filterwarnings('ignore')

            #### ERRONEOUS DETECTION ####
            if degradation == '1':
                self.plot_date = plotting_date

                #### PREPROCESSING ####
                #Check if preprocessed already
                delim = "\\" if "\\" in self.inpath_detection else "/"
                self.subout = list(filter(None, self.inpath_detection.split(delim)))[-1]
                self.run_preprocessing()
                self.datestring = [x for x in os.listdir(self.outpath + self.subout + "/processed data/") if 'neighbors' in x][0][-29:-5]

                #### ANALYSIS ####
                self.run_analysis()

                #### PLOTS ####
                self.run_plotting("Plots already exists, regenerate plots? (y/n): ")

            else:
                #### DEGRADATION ####
                delim = "\\" if "\\" in self.inpath_degradation else "/"
                self.subout = list(filter(None, self.inpath_degradation.split(delim)))[-1]

                if degradation == '3':
                    df_fixed_sensors = get_fixed(self.outpath)
                    self.run_degradation(fixed_sensors=df_fixed_sensors)
                else:
                    self.run_degradation()

                # #### REPLOTS ####
                # self.run_plotting("Do you want to update the plots with degradation results? (y/n): ")

            while exit != "y" and exit != "n":
                exit = input("Analysis Complete, exit? (y/n): ")

    def run_preprocessing(self):
        the_path = self.outpath + self.subout + '/processed data'
        if os.path.isdir(the_path) and len(os.listdir(the_path)) > 0:
            pp = ''
            # Asks if you want to run it again
            while any([pp is 'n', pp is 'y']) is False:
                pp = input("Processed data already exists for " + self.subout +
                           ". Do you want to run pre-processing again (y/n)?: ")
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
                    PlotsToDocx(inpath=self.inpath_detection,
                                outpath=self.outpath + self.subout,
                                plot_date=self.plot_date,
                                date_range_string=self.datestring
                                ).save()

        # Runs it if it doesn't exist
        else:
            PlotMisconfigs(inpath=self.inpath_detection,
                           outpath=self.outpath + self.subout,
                           plot_date=self.plot_date,
                           date_range_string=self.datestring)

            PlotsToDocx(inpath=self.inpath_detection,
                        outpath=self.outpath + self.subout,
                        plot_date=self.plot_date,
                        date_range_string=self.datestring
                        ).save()

    def run_degradation(self, fixed_sensors=None):
        the_path = self.outpath + self.subout
        degtype = 'fixed' if isinstance(fixed_sensors, pd.DataFrame) else 'all'

        if os.path.isdir(the_path) and any([degtype in x for x in os.listdir(the_path)]):
            pp = ''
            # Asks if you want to run it again
            while any([pp is 'n', pp is 'y']) is False:
                pp = input("Degradation analysis already exist, run it again? (y/n): ")
                if len(pp) > 1:
                    pp = pp[0].lower()
                if pp is "y":
                    GetDegradation(inpath=self.inpath_degradation,
                                   outpath=self.outpath + self.subout,
                                   bad_sensors=fixed_sensors,
                                   saved=False)
        # Runs it if it doesn't exist
        else:
            GetDegradation(inpath=self.inpath_degradation,
                           outpath=self.outpath + self.subout,
                           bad_sensors=fixed_sensors,
                           saved=False)
