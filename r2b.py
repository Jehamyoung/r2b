import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import argparse
from skopt import BayesSearchCV


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="XGBoost Model with Feature Selection, Clustering, and Bayes Optimization")
    parser.add_argument('csv_file', type=str, help="Path to the input CSV file")
    parser.add_argument('start_col', type=int, help="Start column index for features (1-based)")
    parser.add_argument('end_col', type=int, help="End column index for features (1-based)")
    parser.add_argument('k', type=int, help="Number of clusters for KMeans")
    parser.add_argument('--bayes_opt', action='store_true', help="Use Bayesian optimization for hyperparameter tuning")
    args = parser.parse_args()

    # Load the data
    df = pd.read_csv(args.csv_file)

    # Get feature columns based on the provided column range
    features = df.columns[args.start_col - 1:args.end_col]  # Convert to 0-based index

    # Separate target variable and features
    X = df[list(features)]
    y = df.iloc[:, 1]  # Assuming the target is in the second column

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply KMeans clustering based on the specified number of clusters
    kmeans = KMeans(n_clusters=args.k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # Add cluster labels to the dataframe
    df['Cluster'] = clusters

    # Store new features (coefficients from regression)
    new_features = []

    # Perform linear regression within each cluster
    for cluster_id in range(args.k):
        cluster_data = df[df['Cluster'] == cluster_id]

        # Independent variables are the features, dependent variable is the target (y)
        X_cluster = cluster_data[list(features)]
        y_cluster = cluster_data.iloc[:, 1]

        # Fit a linear regression model
        model = LinearRegression()
        model.fit(X_cluster, y_cluster)

        # Get regression coefficients
        coefficients = model.coef_

        # Add regression coefficients and cluster label as new features
        for idx, row in cluster_data.iterrows():
            new_features.append([*row[list(features)], *coefficients])  # Add features and coefficients as new features

    # Convert to a new DataFrame with the updated features
    new_features_df = pd.DataFrame(new_features,
                                   columns=list(features) + [f'Coef{i + 1}' for i in range(len(features))])

    # Merge new features with the target column
    final_data = pd.concat([new_features_df, y], axis=1)

    # Split data into training and testing sets
    X_final = final_data.drop(columns=[y.name])  # Features
    y_final = final_data[y.name]  # Target

    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

    # If Bayesian optimization is enabled
    if args.bayes_opt:
        # Define the hyperparameter space for XGBoost
        param_space = {
            'learning_rate': (0.01, 0.3, 'uniform'),  # Learning rate
            'max_depth': (3, 12),  # Maximum depth of trees
            'subsample': (0.6, 1.0),  # Subsample ratio
            'colsample_bytree': (0.6, 1.0),  # Column sample ratio
            'n_estimators': (50, 200),  # Number of trees
        }

        # Create an XGBoost classifier
        model_xgb = xgb.XGBClassifier(eval_metric='logloss')

        # Use Bayesian optimization for hyperparameter search
        opt = BayesSearchCV(model_xgb, param_space, n_iter=50, random_state=42, cv=3, verbose=0)
        opt.fit(X_train, y_train)

        # Get the best hyperparameters
        best_params = opt.best_params_
        print("Best Hyperparameters: ", best_params)

        # Train the model using the best hyperparameters
        model_xgb = xgb.XGBClassifier(**best_params, eval_metric='logloss')
        model_xgb.fit(X_train, y_train)

    else:
        # If Bayesian optimization is not enabled, use default hyperparameters
        model_xgb = xgb.XGBClassifier(eval_metric='logloss')
        model_xgb.fit(X_train, y_train)

    # Make predictions
    y_pred = model_xgb.predict(X_test)
    y_prob = model_xgb.predict_proba(X_test)[:, 1]  # Get predicted probabilities for AUC

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Calculate AUC
    auc = roc_auc_score(y_test, y_prob)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    TP = cm[1, 1]  # True positive
    FN = cm[1, 0]  # False negative

    # Calculate False Negative Rate (FNR)
    if (TP + FN) > 0:
        FNR = FN / (TP + FN)
    else:
        FNR = 0  # Avoid division by zero

    # Store results in a dictionary
    results = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'AUC': auc,
        'FNR': FNR,
        'Clusters': args.k,
        'Features': ', '.join(features)
    }

    # Convert the results to a DataFrame and save to a CSV file
    results_df = pd.DataFrame([results])
    results_df.to_csv('output.csv', index=False)

    print("results saved to 'output.csv'.")


if __name__ == '__main__':
    main()