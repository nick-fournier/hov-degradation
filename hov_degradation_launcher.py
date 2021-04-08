from hov_degradation.__main__ import main

if __name__ == '__main__':
    # main()
    # main(detection_data_path="experiments/input/D7/5min/2020_12_06-12/",
    #      degradation_data_path="experiments/input/D7/hourly/2019/",
    #      output_path="experiments/output/",
    #      plotting_date="2020-12-09"  # This is a wednesday
    #      )
    main(detection_data_path="experiments/input/D7/5min/2021_03_01-07/",
         degradation_data_path="experiments/input/D7/hourly/2020/",
         output_path="experiments/output/",
         plotting_date="2021-03-03"  # This is a wednesday
         )