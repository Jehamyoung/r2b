# r2b: Regression to Boost

r2b (Regression to Boost) is a machine learning model designed to address the challenge of having limited feature dimensions in the data. When the number of features is small, it can be difficult for traditional machine learning models to perform effectively. r2b enhances predictive power by utilizing a combination of  **KMeans Clustering**, **Linear Regression**, and **XGBoost** to create a robust pipeline for classification tasks.

The model leverages clustering and regression within each cluster to generate new features, which are then used to train the XGBoost classifier. This approach improves the model's ability to learn from smaller datasets with fewer features.

## Features

- **KMeans Clustering**: Clusters data based on the selected features to generate subgroups for further analysis.
- **Linear Regression**: Applies linear regression within each cluster to generate new features (coefficients).
- **XGBoost**: Trains an XGBoost model on the newly created features and evaluates its performance.
- **Bayesian Optimization**: Optionally optimizes hyperparameters for the XGBoost model using Bayesian Optimization.
- **Model Evaluation**: Computes and saves key evaluation metrics: Accuracy, Precision, Recall, F1-Score, AUC, and False Negative Rate (FNR).


## Requirements

To run this project, you need Python 3.6+ and the following libraries:

- pandas
- numpy
- scikit-learn
- xgboost
- scikit-optimize (for Bayesian optimization)

You can install the necessary libraries by running:

```bash
pip install pandas numpy scikit-learn xgboost scikit-optimize
```

## Easy Usage

### Data Form

**The target is in the second column!**

| name       | Category | ... | Feature_n |
|------------|------------|------------|----------|
| A   | 0     | ...          | 20     |
| B | 1      | ...          | 30      |
| ...  | ...        | ...        |    ...   |
| C   | 1       | ...          | 53      |


### Command-Line Arguments

The script can be run from the command line with the following arguments:
#### Note that the index of the first column starts from 1, not 0

1. **`csv_file`** (required): The path to the input CSV file containing your data.
2. **`start_col`** (required): The starting column index for feature selection (1-based).
3. **`end_col`** (required): The ending column index for feature selection (1-based).
4. **`k`** (required): The number of clusters for KMeans clustering.
5. **`--bayes_opt`** (optional): Flag to enable Bayesian optimization for hyperparameter tuning.

### Example Command

To run the script without Bayesian optimization:

```bash
python r2b.py data.csv 3 6 4
```

To run the script with Bayesian optimization ():

```bash
python r2b.py data.csv 3 6 4 --bayes_opt
```

## How to Cite

If you use this code in your research or paper, please cite it as follows:

**Yang, Jiahuan, Junxiang Huang, and Lai Jiang. "Reg2Boost: A Machine Learning Optimization Approach for Low-Dimensional Breast Cancer Classification." 2025 8th International Symposium on Big Data and Applied Statistics (ISBDAS). IEEE, 2025.**
