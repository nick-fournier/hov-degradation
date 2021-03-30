# hov-degradation
Serves as the repository for HOV-degradation analysis

## Contents

* [Setup Instructions](#setup-instructions)
* [Usage](#usage)

## Setup Instructions
If you have not done so already, download the hov-degradation github repository.

```bash
git clone https://github.com/nick-fournier/hov-degradation
cd hov-analysis
```

Next, create the Conda environment and install hov-degradation and its
dependencies within the environment. You can install Conda from
[Anaconda](https://www.anaconda.com/download)

```bash
conda env create -f environment.yml
source activate hov-analysis
python setup.py develop
```

If the setup.py installations fails, install the contents using pip

```bash
pip install -e .
```

## Usage


#### 1. util/data_prerocess.py
This preprocesses the data for training. Requires file path and the start/end dates for the data.
   
#### 2. train/train.py
Trains the data and then tests it for I-210, and runs it for entire D7.

#### 3. util/plot.py
Plots the identified misconfigurations in D7.

4. agg_results.py 
5. degraded.py


