from hov_degradation.__main__ import main
import os

if __name__ == '__main__':
    """
    Compile into single exe using this command in bash:
    pyinstaller -F hov_degradation_launcher.py
    """

    # Checking config file
    if os.path.isfile("hov_degradation_config.txt"):
        config = {}
        with open("hov_degradation_config.txt") as fh:
            for line in fh:
                line_list = line.strip().split()
                if len(line_list) > 0:
                    var, value = line_list
                    config[var] = value.strip()

        for p in ['detection_data_path', 'degradation_data_path']:
            if not (os.path.isdir(config[p]) and len(os.listdir(config[p]))):
                config[p] = None
    else:
        config = {'detection_data_path': None,
                  'degradation_data_path': None,
                  'output_path': None,
                  'plotting_date': None}

    # Running entry script
    main(detection_data_path=config['detection_data_path'],
         degradation_data_path=config['detection_data_path'],
         output_path=config['output_path'],
         plotting_date=config['plotting_date']
         )
