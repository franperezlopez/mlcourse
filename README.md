Machine Learning Introduction 
=============================

Install using local Environment

Prerequisites:

* Conda installation

You need to execute following commands to create a new python environment, containing all libraries needed for the course:

```sh
conda update -n base -c defaults conda
conda env create --name mlcourse -f environment.yml
conda activate mlcourse
python -m ipykernel install --user --name mlcourse --display-name "Python (mlcourse)"
jupyter notebook
```

After executing last command, the jupyter notebook will get displayed in your browser. The course notebooks are located in the folder named `notebooks`. After opening any notebook, check that the kernel you're using is the one named `Python (mlcourse)`, otherwise change it using the menu `Kernel` -> `Change kernel` -> `Python (mlcourse)`.


Install using Azure Notebook

Import this repo to you Azure Notebook account [![Azure Notebooks](https://notebooks.azure.com/launch.png)](https://notebooks.azure.com/import/gh/franperezlopez/mlcourse)

Clone this project into your account 


Project Organization
--------------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks
    │   └── ml1.ipynb      <- Machine Learning Course : Random Forest
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── utils.py       <- Helper methods for handling data and models
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   ├── models         <- Scripts to train models and then use trained models to make predictions
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
