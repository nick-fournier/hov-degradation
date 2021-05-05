"""
Copyright ©2021. The Regents of the University of California (Regents). All Rights Reserved. Permission to use, copy, modify, and distribute this software and its documentation for educational, research, and not-for-profit purposes, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright notice, this paragraph and the following two paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
import sys

from hov_degradation.__main__ import main
import os

if __name__ == '__main__':
    """
    Compile into single exe using this command in bash:
    pyinstaller -F hov_degradation_launcher.py
    """

    # Checking config file
    if os.path.isfile('./dist/hov_degradation_config.txt'):
        config_path ='./dist/hov_degradation_config.txt'
    elif os.path.isfile('hov_degradation_config.txt'):
        config_path = 'hov_degradation_config.txt'
    else:
        config_path = None


    if os.path.isfile(config_path):
        config = {}
        with open(config_path) as fh:
            for line in fh:
                line_list = line.strip().split(',')
                if len(line_list) > 0:
                    var, value = line_list
                    config[var] = value.strip()

        print("=========================================================")
        print("Current directories:")
        for p in ['detection_data_path', 'degradation_data_path']:
            print(p + ": " + config[p])
            if not (os.path.isdir(config[p]) and len(os.listdir(config[p])) > 0):
                config[p] = None
                input('Cannot find data files for "' + p + '" Press any key to continue')

        if not os.path.isdir(config['output_path']):
            os.makedirs(config['output_path'])

        if not any([config['plotting_date'].replace('-', '_') in x for x in os.listdir(config['detection_data_path'])]):
            config['plotting_date'] = None
            input('Plot date ' + config['plotting_date'] + ' not found in ' +
                  config['detection_data_path'] + ' Press any key to continue')

        if not all(config.values()):
            sys.exit()

        # Running entry script
        print("=========================================================")
        main(detection_data_path=config['detection_data_path'],
             degradation_data_path=config['degradation_data_path'],
             output_path=config['output_path'],
             plotting_date=config['plotting_date']
             )

    else:
        input('Cannot find configuration file "hov_degradation_config.txt". '
              'It should be located in same folder as the .exe. '
              'Press any key to continue.')
