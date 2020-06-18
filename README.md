# hov-degradation
Serves as the repository for HOV-degradation analysis

## Contents

* [Setup Instructions](#setup-instructions)

## Setup Instructions
If you have not done so already, download the hov-degradation github repository.

```bash
git clone https://github.com/Yasharzf/hov-degradation.git
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