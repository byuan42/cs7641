# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 12:55:31 2024

@author: boyua
"""
import pandas as pd
import geopandas as gp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import os
import zipfile
# Extract the zip file
zip_path = "limited_data.zip"
extract_path = "."

if not os.path.exists(extract_path + "/limited_data.csv"):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# List the files to ensure extraction
limited_data_file = os.path.join(extract_path, "limited_data.csv")
# Load the data
limited_data = pd.read_csv(limited_data_file)
limited_data_prcp_date_station = limited_data[limited_data['Data type'] == 'PRCP'][['Date', 'Station']]
limited_data_prcp_filter = pd.merge(limited_data,limited_data_prcp_date_station,on=['Date', 'Station'])
limited_data_tmax_date_station = limited_data[limited_data['Data type'] == 'TMAX'][['Date', 'Station']]
limited_data_tmax_filter = pd.merge(limited_data_prcp_filter,limited_data_tmax_date_station,on=['Date', 'Station'])
limited_data_tobs_date_station = limited_data[limited_data['Data type'] == 'TOBS'][['Date', 'Station']]
limited_data_tobs_filter = pd.merge(limited_data_tmax_filter,limited_data_tobs_date_station,on=['Date', 'Station'])
data = pd.concat([limited_data_tobs_filter[limited_data_tobs_filter['Data type']=='TOBS']['Value'].reset_index().rename(columns={'index':'i1','Value':'TOBS'}),limited_data_tobs_filter[limited_data_tobs_filter['Data type']=='TMAX']['Value'].reset_index().rename(columns={'index':'i2','Value':'TMAX'}),limited_data_tobs_filter[limited_data_tobs_filter['Data type']=='PRCP']['Value'].reset_index().rename(columns={'index':'i3','Value':'PRCP'})],axis=1)
data.dropna(inplace=True)
train_x,test_x,train_y,test_y = train_test_split(StandardScaler().fit_transform(data[['TOBS','PRCP']]), data['TMAX'], test_size=0.2, random_state=0)
# Train and evaluate models
def evaluate_model(predictions, true_values):
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    return mae, mse, r2

# Neural Network
nn_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
nn_model.fit(train_x, train_y)
nn_preds = nn_model.predict(test_x)
nn_mae, nn_mse, nn_r2 = evaluate_model(nn_preds, test_y)

# SVM with RBF Kernel
svm_model_rbf = SVR(kernel='rbf')
svm_model_rbf.fit(train_x, train_y)
svm_rbf_preds = svm_model_rbf.predict(test_x)
svm_rbf_mae, svm_rbf_mse, svm_rbf_r2 = evaluate_model(svm_rbf_preds, test_y)

# SVM with Linear Kernel
svm_model_linear = SVR(kernel='linear')
svm_model_linear.fit(train_x, train_y)
svm_linear_preds = svm_model_linear.predict(test_x)
svm_linear_mae, svm_linear_mse, svm_linear_r2 = evaluate_model(svm_linear_preds, test_y)

# k-NN
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(train_x, train_y)
knn_preds = knn_model.predict(test_x)
knn_mae, knn_mse, knn_r2 = evaluate_model(knn_preds, test_y)

print(f'Neural Network - MAE: {nn_mae}, MSE: {nn_mse}, R2: {nn_r2}')
print(f'SVM (RBF) - MAE: {svm_rbf_mae}, MSE: {svm_rbf_mse}, R2: {svm_rbf_r2}')
print(f'SVM (Linear) - MAE: {svm_linear_mae}, MSE: {svm_linear_mse}, R2: {svm_linear_r2}')
print(f'k-NN - MAE: {knn_mae}, MSE: {knn_mse}, R2: {knn_r2}')


# Example of Grid Search for SVM
param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf', 'linear']}
grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=2)
grid.fit(train_x, train_y)

print("Best Parameters:", grid.best_params_)
print("Best Estimator:", grid.best_estimator_)

def plot_learning_curve(model, X_train, y_train, X_test, y_test, title):
    train_sizes = np.linspace(0.1, 0.9, 10)
    train_scores = []
    test_scores = []

    for train_size in train_sizes:
        X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=42)
        model.fit(X_train_subset, y_train_subset)
        train_scores.append(model.score(X_train_subset, y_train_subset))
        test_scores.append(model.score(X_test, y_test))

    plt.figure()
    plt.plot(train_sizes, train_scores, label='Training Score')
    plt.plot(train_sizes, test_scores, label='Test Score')
    plt.title(title)
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

plot_learning_curve(nn_model, train_x, train_y, test_x, test_y, "Neural Network Learning Curve")
plot_learning_curve(svm_model_rbf, train_x, train_y, test_x, test_y, "SVM (RBF) Learning Curve")
plot_learning_curve(knn_model, train_x, train_y, test_x, test_y, "k-NN Learning Curve")