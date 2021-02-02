# hov-degradation
Serves as the repository for HOV-degradation analysis

## Contents

* [Setup Instructions](#setup-instructions)

## Setup Instructions
If you have not done so already, download the hov-degradation github repository.

```bash
git clone https://git-codecommit.us-west-2.amazonaws.com/v1/repos/hov-degradation
cd hov-degradation
```

Next, create the Conda environment and install hov-degradation and its
dependencies within the environment. You can install Conda from
[Anaconda](https://www.anaconda.com/download)

```bash
conda env create -f environment.yml
source activate hov-degradation
python setup.py develop
```

If the setup.py installations fails, install the contents using pip

```bash
pip install -e .
```

## Usage
#### 1. data_prerocess.py
   This preprocesses the data for training. Requires path to the data and the start/end dates of the data.
   
2. train.py
3. plot.py 
4. agg_results.py 
5. degraded.py


