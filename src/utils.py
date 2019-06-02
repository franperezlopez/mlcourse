import pandas as pd
import math
from sklearn.tree import export_graphviz
import IPython, graphviz, re #, sklearn_pandas, sklearn, warnings, pdb
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import forest
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

#for type annotations
from numbers import Number
from typing import Any, AnyStr, Callable, Collection, Dict, Hashable, Iterator, List, Mapping, NewType, Optional
from typing import Sequence, Tuple, TypeVar, Union
from types import SimpleNamespace

def display_all(df:pd.DataFrame, max_rows = 1000, max_columns = 1000):
    """Display more rows and columns
       up to max_rows rows and max_columns columns
    Arguments:
        df {pd.DataFrame} -- data to be displayed
    Keyword Arguments:
        max_rows {int} -- max rows to display (default: {1000})
        max_columns {int} -- max columns to display (default: {1000})
    """
    with pd.option_context("display.max_rows", max_rows, "display.max_columns", max_columns):
        display(df)

def split_vals(df: pd.DataFrame, size: int) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """Splits dataframe by size, returning two dataframes.
    First dataframe has length = size, and second dataframe has length = len(pd) - size
    Arguments:
        df {pd.DataFrame} -- dataframe
        size {int} -- size of the first dataframe
    Returns:
        Tuple[pd.DataFrame,pd.DataFrame] -- first dataframe, second dataframe
    """
    return df[:size].copy(), df[size:].copy()

def rmse(y_hat: List[Number],y: List[Number]):
    """Calculates RMSE metric
    
    Arguments:
        y_hat {float} -- prediction
        y {float} -- ground truth
    
    Returns:
        float -- RMSE value
    """
    return math.sqrt(((y_hat-y)**2).mean())

def add_datepart(df: pd.DataFrame, dateFieldName:str, dropDateField:bool=True, time:bool=False):
    """Helper function that adds columns relevant to a date.
    
    Arguments:
        df {pd.DataFrame} -- dataframe
        dateFieldName {str} -- datetime field name
    
    Keyword Arguments:
        dropDateField {bool} -- remove original datetime field from dataset (default: {True})
        time {bool} -- extract time detail (default: {False})
    """
    fld = df[dateFieldName]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[dateFieldName] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', dateFieldName)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Epoch'] = fld.astype(np.int64) // 10 ** 9
    if dropDateField: df.drop(dateFieldName, axis=1, inplace=True)

def draw_tree(t, df, size=10, ratio=0.6, precision=0):
    """ Draws a representation of a random forest in IPython.
    Parameters:
    -----------
    t: The tree you wish to draw
    df: The data used to train the tree. This is used to get the names of the features.
    """
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,
                      special_characters=True, rotate=False, precision=precision)
    IPython.display.display(graphviz.Source(re.sub('Tree {', f'Tree {{ size={size}; ratio={ratio}', s)))

def df_pre(df:pd.DataFrame, inplace:bool = False):
    """Feature Engineering on dataset
    
    Arguments:
        df {pd.DataFrame} -- dataframe
    
    Keyword Arguments:
        inplace {bool} -- transforms original dataset (default: {False})
    
    Returns:
        [type] -- transformed dataframe
    """
    if not inplace:
        df = df.copy()

    columns_cat = df.select_dtypes('object').head(0).columns.values.tolist()
    for col in columns_cat:
        df[col] = df[col].astype('category').cat.codes + 1

    columns_null = (df.isnull().sum()>0)[lambda m: m==True].index.tolist()
    for col in columns_null:
        df[col+'_na'] = df[col].isna()
        df[col].fillna(df[col].median(), inplace=True)

    return df


def dectree_max_depth(tree):
    """Calculates decission tree depth
    
    Arguments:
        tree {decission tree} -- Decission tree
    
    Returns:
        int -- decission tree depth
    """
    children_left = tree.children_left
    children_right = tree.children_right

    def walk(node_id):
        if (children_left[node_id] != children_right[node_id]):
            left_max = 1 + walk(children_left[node_id])
            right_max = 1 + walk(children_right[node_id])
            return max(left_max, right_max)
        else: # leaf
            return 1

    root_node_id = 0
    return walk(root_node_id)


def plot_learning_curve(m, X, y, plot_training = True, plot_validation= True, plot_ci = True):
    """Plot model learning curve
    
    Arguments:
        m {[type]} -- sklearn model
        X {[type]} -- features dataframe
        y {[type]} -- label dataframe
    
    Keyword Arguments:
        plot_training {bool} -- plot training curve (default: {True})
        plot_validation {bool} -- plot validation curve (default: {True})
        plot_ci {bool} -- plot confidence interval (default: {True})
    """
    scoring = 'r2'
    train_sizes, train_scores, test_scores = learning_curve(m,
                                                        X,
                                                        y,
                                                        cv=3, # Number of folds in cross-validation
                                                        scoring=scoring, # Evaluation metric
                                                        n_jobs=-1, # Use all computer cores
                                                        train_sizes=np.linspace(0.01, 1.0, 50) # 50 different sizes of the training set
                                                        #train_sizes=[0.01, 0.05]
                                                        )

    if plot_training:
        # Create means and standard deviations of training set scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)

        # Draw lines
        plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")

        if plot_ci:
            # Draw bands
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")

    if plot_validation:
        # Create means and standard deviations of test set scores
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Draw lines
        plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

        if plot_ci:
            # Draw bands
            plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel(f"Score ({scoring})"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

def set_rf_samples(n):
    """ Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))

def reset_rf_samples():
    """ Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))


def categorify(df, inplace=True):
    columns_cat = df.select_dtypes('object').head().columns.values.tolist()
    df_target = df if inplace else df.copy()
    for col in columns_cat:
        df_target[col] = df[col].astype('category').cat.codes
    return df_target

def imputation(df, inplace=True):
    return df.fillna(df.median(), inplace=inplace)

def set_plot_sizes(sml, med, big):
    plt.rc('font', size=sml)          # controls default text sizes
    plt.rc('axes', titlesize=sml)     # fontsize of the axes title
    plt.rc('axes', labelsize=med)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=sml)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=sml)    # fontsize of the tick labels
    plt.rc('legend', fontsize=sml)    # legend fontsize
    plt.rc('figure', titlesize=big)  # fontsize of the figure title

def parallel_trees(m, fn, n_jobs=8):
        return list(ThreadPoolExecutor(n_jobs).map(fn, m.estimators_))

def df_rmse(m, df, y):
    y_hat = m.predict(df)
    return math.sqrt(np.sum((y_hat - y)**2)/len(y))