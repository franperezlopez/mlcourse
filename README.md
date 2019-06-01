## Machine Learning Introduction 


### Install using local Environment

#### Prerequisites:

* Conda installation: [Install Anaconda](http://docs.anaconda.com/anaconda/install/) for your OS

* After installing Conda, you need to execute following commands from the folder where the repository is located:
```sh
conda update -n base -c defaults conda
conda env create --name mlcourse -f environment.yml
conda activate mlcourse
python -m ipykernel install --user --name mlcourse --display-name "Python (mlcourse)"
jupyter notebook
```
The previous script will create a new python environment for the course, with all packages needed to run the notebooks.



### Install using Azure Notebook

Visit the notebook [mlcourse](https://notebooks.azure.com/franperez/projects/mlcourse) in Azure Notebook, and click on the Clone icon to clone the project to your Azure Notebook workspace

Alternatively, you can import this repo to you Azure Notebook account [![Azure Notebooks](https://notebooks.azure.com/launch.png)](https://notebooks.azure.com/import/gh/franperezlopez/mlcourse)

Once the project is in your workspace, click on the Button `Run on Free Computer` to start the container; after the container is initializer, you will get the usual Jupyter Notebook screen page.

### Execute the notebook

After executing Jupyter server, you can check the notebook located in the folder named `notebooks`. After opening any notebook, check that the kernel you're using is the one named `Python (mlcourse)`, otherwise change it using the menu `Kernel` -> `Change kernel` -> `Python (mlcourse)`.

Bear in mind, executing the Notebook using Azure Notebook is **slow** as this is a free service. You will get a better performance executing the notebook in your local machine, or executing the Azure Notebook in a server not from the free tier.



### Project Organization

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
