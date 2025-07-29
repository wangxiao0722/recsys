# RecSys
A hands-on tutorial project for classic recommendation algorithms, from Logistic Regression to Transformer, implemented with PyTorch.

## Environment Setup
Requires Python 3.7+ and highly recommended to use Anaconda for environment and dependency management.

An `environment.yml` file is provided in the root directory containing all necessary packages.

## Quick Start
### 1. Clone repository and setup environment
```bash
git clone https://github.com/wangxiao0722/recsys.git
cd recsys
conda env create -f environment.yml
conda activate recsys-env
```
### 2. Run the logistic regression model
```bash
python python models/lr.py
```
### 3. Verify results
```bash
head data/result.csv
```

## Step-by-Step Tutorials
For detailed walkthroughs of each algorithm with executable examples, visit our project wiki:

[📚 RecSys Wiki Homepage](https://github.com/wangxiao0722/recsys/wiki)