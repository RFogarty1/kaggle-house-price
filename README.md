# Kaggle House Prices Prediction

These are some notebooks associated with the Kaggle ["House Prices - Advanced Regression Techniques"](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition.

# Structure

Notebooks are commited in a "run" state in three folders:

**eda** - "Exploratory Data Analysis". Mainly focused on mutual information between features and the target.

**model** - Notebooks showing effects of changing features and hyperparamters on random forest/XGBoost trees performances

**submission** - Contains a single notebook which writes a file which can be submitted to Kaggle


# Setup

Notebooks are better seen in the Jupyter environment vs directly on GitHub, since theres a lot of (false positive) warnings from Pandas which can be largely hidden in Jupyter.

Assuming a linux OS with pyenv installed, the bash commands to setup a python virtual environment with the required dependencies can be found in install\_deps.sh.

Furthermore, the required competition data needs to be moved into the raw\_data folder. This data can be found [here](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data). For example, the training data should be located at "raw\_data/train.csv" (relative to the base directory).


