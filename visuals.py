###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score


def evaluate(results):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
    """

    # Create figure
    fig, ax = pl.subplots(2, 2, figsize=(11, 11))

    # Constants
    bar_width = 0.3
    colors = '#00A0A0'
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'r2_train', 'pred_time', 'r2_val']):

            # Creative plot code
            ax[j / 2, j % 2].bar(1.1 * k * bar_width, results[learner][metric], width=bar_width, color=colors)
            ax[j / 2, j % 2].set_xticks([0.33 * x for x in range(len(results.keys()))])
            ax[j / 2, j % 2].set_xticklabels(['Boosting', 'LR', 'ET', 'DT', 'Bagging', 'KNN', 'Linear', 'AdaB', 'RF', 'SVM'])
            ax[j / 2, j % 2].set_xlabel("Common Regressor")

    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("R2 Score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("R2 Score")

    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("R2 Score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("R2 Score on Validation Set")

    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))

    # Aesthetics
    pl.suptitle("Performance Metrics for Common Supervised Learning Models", fontsize=16, y=1.10)
    pl.tight_layout()
    pl.show()
