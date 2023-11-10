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

# How to run Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

# How to visualize Kedro pipeline

You can visualize Kedro project with:

```
kedro viz
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


### How to convert notebook cells to nodes in a Kedro project
You can move notebook code over into a Kedro project structure using a mixture of [cell tagging](https://jupyter-notebook.readthedocs.io/en/stable/changelog.html#cell-tags) and Kedro CLI commands.

By adding the `node` tag to a cell and running the command below, the cell's source code will be copied over to a Python file within `src/<package_name>/nodes/`:

```
kedro jupyter convert <filepath_to_my_notebook>
```
> *Note:* The name of the Python file matches the name of the original notebook.

Alternatively, you may want to transform all your notebooks in one go. Run the following command to convert all notebook files found in the project root directory and under any of its sub-folders:

```
kedro jupyter convert --all
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can run `kedro activate-nbstripout`. This will add a hook in `.git/config` which will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

If you want to use MakinaRocks for visualization of pipeline inside the notebook you can run

```
python3 -m pip install mrx-link
```

After installation, enter the following code to execute Jupyter Lab.
```
python3 -m jupyterlab
```

Open Notebook register in Makina Rocks and explore the pipeline inside :)

