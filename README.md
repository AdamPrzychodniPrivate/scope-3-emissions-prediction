# Scope 3 Emissions Prediction - in progress :wrench:

## Overview

This project is inspired by the academic research conducted by George Serafeim and Gladys Vélez Caicedo of Harvard Business School. The paper, titled
[Machine Learning Models for Prediction of Scope 3 Carbon Emissions](https://www.hbs.edu/ris/Publication%20Files/22%20080_035d70d9-3acf-4faa-aa93-534e52a52d0e.pdf) serves as a foundational framework for my analysis.

Leveraging the [dataset](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiEmPzH_IGBAxXH_rsIHV1ZB8kQFnoECA4QAQ&url=https%3A%2F%2Fwww.hbs.edu%2Fimpact-weighted-accounts%2FDocuments%2FIWA-External-Scope-3-Data.xlsx&usg=AOvVaw2HzIPtLsvHCWiHQKA5o_m8&opi=89978449) provided in the research, my objective is to explore various modeling techniques to accurately predict the value of Scope 3 Carbon Emissions.

## How to create a new virtual environment using conda

The recommended approach. From your terminal:
```
conda create --name kedro-environment python=3.10 -y
```

Activate your environment
```
conda activate kedro-environment
```

## How to install dependencies

Declare any dependencies in `src/requirements.txt` for `pip` installation and `src/environment.yml` for `conda` installation.

To install them, run:

```
pip install -r src/requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
kedro test
```

To configure the coverage threshold, go to the `.coveragerc` file.


### Jupyter

You can use Jupyter Notebook:

```
kedro jupyter notebook
```

### JupyterLab

Or Jupyter Lab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```
