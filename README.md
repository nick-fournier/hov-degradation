# HOV Degradation
Serves as the repository for HOV-degradation analysis

## Contents

* [Setup Instructions](#setup-instructions)
* [Usage](#usage)

## Setup Instructions
To run the code, you can either download and run the standalone the executable file, or install the python source code as a developer.

#### Executable
To run the executable:
1. Download the executable from `dist/hov_degradation_launcher.exe`,
2. Create input and output folders for 5-minute and hourly data 
3. Specify input/output paths in the `hov_degradation_config.txt` file.
4. Download the necessary data (e.g., from [PeMS](https://pems.dot.ca.gov/) data clearinghouse at https://pems.dot.ca.gov/) 
   1. Detection data: 5-minute count data (recommend at least 5 weekdays of data, minimum of 24-hours)
   2. Degradation data: hourly count data (180 days minimum per FHWA requirements) 
   3. Meta-data: choose date nearest the date of data downloaded, place one in each 5-minute and hourly data folder 
3. Run the executable, if set up correctly, you will be prompted to run:
   1. Erroneous HOV detection on 5-minute traffic data
   2. Degradation analysis on hourly traffic data
   3. Magnitude of erroneous degradation (requires HOV lane corrections)

*Note that it might take a few seconds for the executable to start. 

Here is an example file structure to run the program:
```
HOV degradation main directory
├─ hov_degradation_launcher.exe
├─ hov_degradation_config.txt
├─ input
│   ├─ 5min_counts
│   │   ├─ d07_text_meta_2020_11_16.txt
│   │   └─ d07_text_station_5min_2020_12_12.txt.gz
│   └─ hourly_counts
│       ├─ d07_text_meta_2020_11_16.txt    
│       └─ d07_text_station_hour_2020_01.txt.gz    
└─ output
```

#### Developer Setup
Clone the hov-degradation github repository.

```bash
git clone https://github.com/nick-fournier/hov-degradation
cd hov-analysis
```

Next, set up a virtual environment to run this package and its dependencies. This can be achieved using a virtual environment of your choice, such as Python's venv or Conda. We're going to use Conda in this example. 

In the bash terminal, create the Conda environment and activate it, then install hov-degradation and its dependencies within the environment. You can install Conda from:
[https://www.anaconda.com/download](https://www.anaconda.com/download)

```bash
conda env create -f environment.yml
source activate hov-analysis
python setup.py develop
```

If the setup.py installations fails, you can install the contents using pip from this directory.

```bash
pip install -e .
```

The point of entry for the program is `hov_degradation_launcher.py` (this is just a pointer to `hov_degradation/__main__.py`). To run the program, simple run the launcher script, then a command prompt will ask you a series of inputs.   

## Usage
To run the program, it requires the file paths for an input and output data folders to direct the code to find. If you are using the `.exe`, you may specify the file paths using the `hov_degradation_config.txt` file. Otherwise the program will default to promting you to type the file paths in manually. 

I've organized my developer directory like this, but you can input custom file paths to any desired input and output locations:
```
HOV degradation main directory
├─ hov_degradation code
│   ├─ preprocess
│   ├─ analysis
│   ├─ reporting
│   └─ static
└─ data
    ├─ input
    │   ├─ 5min_counts
    │   └─ hourly_counts
    └─ output
```
Three main types of data are needed:
* **5-minute daily traffic counts**: used for detection of misconfigured sensors.
* **Hourly traffic counts**: used for the degradation analysis *of* the detected sensors.
* **Meta data**: A meta-data file for sensors is required ***in each folder*** for 5-min and hourly data sources. 

Note that the meta-data file could be the same for both 5-minute and hourly counts, but the code requires a separate meta-data file for each data type. This is for consistency in case the study period varies so that the meta-data file is different. 

The file format for meta data is `.txt` and the file format for the count data is a compressed `.txt.gz` filetype downloaded from PeMS, no extraction necessary. 

For 5-minute data I downloaded 7 consecutive days of 5-minute traffic counts. I found more than 7 days of data and the combined dataset becomes a bit too large. It's technically possible to use more than 7 days, but the benefit is likely negligible.


The code is organized into four overall steps:
1. **Preprocessing**
2. **Detection**
3. **Degradation** 
4. **Reporting**
5. **Post-Analysis -- Magnitude of erroneous degradation change**


### 1. Preprocessing
In this step the "raw" unfiltered daily 5-minute traffic count data are processed by first removing sensors that do not meet the minimum data requirements and then performing feature extraction. The features are average nighttime flow/occupancy and the K-S statistic) of the flow profiles. Intermediate output is saved in a new folder `output/processed` which will contain four new files generated by the code.
```
output/processed data/
    ├─ i210_test_data_<start date>_to_<end date>.csv
    ├─ i210_train_data_<start date>_to_<end date>.csv
    ├─ D<district #>_data_<start date>_to_<end date>.csv
    └─ D<district #>_neighbors_<start date>_to_<end date>.json
```
The three "processed" files contain the extracted feature data from the sensor data, two of the files are for testing and training of the machine learning algorithms using the validated I-210 data and the third is for the district-wide detection using the trained model. The "neighbors" file is a json file containing the data-dictionary relating each HOV sensor to its nearest upstream/downstream HOV sensor and its adjacent mainline sensors.

 
### 2. Analysis
In this step the processed data generated in the previous step are then analyzed in two sub-steps. 

   1. *Training & testing of I-210 data:* All eight machine learning methods are trained and tested on the I-210 data to obtain the tuned hyperparameters and fitted model for each method. Each machine learning method is tested for prediction accuracy, the best performing method is used for supervised and unsupervised learning. 
      
      This step has already been performed and the resulting hyperparameters and models are hardcoded into the program. The models can be re-trained, but that functionality has been disabled in this version. The `Dection()` class-object has parameter `retrain=False`. If set to true, retrained model results are saved to
      ```
      output/trained/
          ├─ analysis_scores_<start date>_to_<end date>.csv
          ├─ analysis_hyperparameters_<start date>_to_<end date>.json
          ├─ trained_classification_I210_<start date>_to_<end date>.pkl
          └─ trained_unsupervised_I210_<start date>_to_<end date>.pkl
      ```
   2. *District-wide detection:* The best models are then selected from the scoring (Random Forest and Local Outlier Factor used as default for now) and run over the entire district-wide sensor data using the tuned hyperparameters. This second step will then generate district-wide results:
      ```
      output/results/
          ├─ predictions_table_D<district #>_<start date>_to_<end date>.csv
          ├─ misconfigs_meta_table_D<district #>_<start date>_to_<end date>.csv
          └─ misconfigs_ids_D<district #>_<start date>_to_<end date>.json
      ```

### 3. Reporting \& Manual Evaluation
To evaluate the impact that the misconfigured sensors had on degradation, the detected sensors must be manually evaluated to determine whether a correction can be determined (e.g., label for HOV lane is swapped with a mainline lane).

The detection results will generate individual plots for each sensor comparing the target HOV sensor to its upstream/downstream neighbors (longitudinal comparison) and its adjacent mainline sensors (lateral comparison). Image files for these plots are saved in a new folder within the `output/results` folder, but to further assist in this evaluation process, a word document assembling all the plots together is generated.

```
`output/results/
    ├─HOV plots_<plot date>.docx                   #Plots assembled in one word doc
    └─plots_misconfigs_<start date>_to_<end date>  #Individual image file for each plot
```

The PeMS database uses strip maps to orient the sensor along the freeway. If a strip map image is available for the respective sensors, they may be placed in the `results/strip_maps` folder and will be included in the docx file.

### 4. Degradation
Two types of degradation analysis can be performed with this software:

1. *Simple degradation analysis of HOV sensors**: This simply calculates the percent degradation for _all_ HOV sensors in the hourly sensor data provided. This automatically calculates the degradation results that would otherwise be done manually in a spreadsheet.
   
2. *Magnitude of degradation comparison of "corrected" sensors*: If erroneous sensors are specified with their corrected lanes, this calculates the percent degradation for both the erroneous and corrected HOV sensors.


#### A. Degradation Analysis for all HOV sensors

Degradation analysis will create two output files:

```
output/
   ├─ all_sensor_degradation_results_D<district #>_<start date>_to_<end date>.csv
   └─ processed_degradation_data_D<district #>.csv
```

The first file, `all_sensor_degradation_results_D<district #>_<start date>_to_<end date>.csv`, is the actual degradation results with percent degradation and other summary calculations (e.g., VHT, VMT, number of days where data are available, etc.). The second file, `processed_degradation_data_D<district #>.csv` is a temporary file containing extracted data for the detected misconfigured sensors. This functions as cached data in case you would like to re-run the degradation on the sensors without having to re-process the hourly data, which is large data file.


#### b. Magnitude of erroneous HOV sensor degradation

After reviewing the detection results, the lane corrections must be placed into a comma separated value (CSV) file titled: `results\fixed_sensors.csv`. The contents of the file take the format:

    ```
    ID,      issue,          real_lane
    717822,  Misconfigured,  Lane 1
    718270,  Misconfigured,  Lane 2
    718313,  Misconfigured,  Lane 1
    762500,  Misconfigured,  Lane 2
    762549,  Misconfigured,  Lane 1
    768743,  Misconfigured,  Lane 4
    769238,  Misconfigured,  Lane 1
    769745,  Misconfigured,  Lane 3
    774055,  Misconfigured,  Lane 1
    ```
where `ID` is the detected misconfigured HOV sensor ID, `issue` is a descriptor field, and `real_lane` is the corrected maineline lane. There is no need to provide a specific corrected sensor ID here, the ID will be pulled from the `neighbors` file. After creating this file, run the code again and follow the prompts to run degradation.

Degradation analysis will again create an additional output files:

```
output/results/
   └─ fixed_degradation_D<district #>_<start date>_to_<end date>.csv
```

The `fixed_degradation_D<district #>_<start date>_to_<end date>.csv` file are the degradation results, similar to simple degradation analysis, but with additional rows showing both the erroneous and corrected sensor results.
