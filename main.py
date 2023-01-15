
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Import the dataset

df = pd.read_csv('Data_for_UCI_named.csv')
print('Dataset size: ', df.shape)

#Delete last column because it contains labels, not needed in this Regression problem
df.drop('stabf', axis=1, inplace=True)

########################################################################################################################
#                                               Exploratory Data Analysis                                              #
########################################################################################################################

#EDA was done via Jupyter Notebook (the .ipynb file is contained in the folder)

########################################################################################################################
#                                               Preprocessing                                                          #
########################################################################################################################

# Remove features which have a correlation coefficient with the output less than a certain threshold

corrmatrix = df.corr()
threshold = 0.1
features_selected = []
corr_selection = corrmatrix['stab']

for i, index in enumerate(corr_selection.index):
    if abs(corr_selection[index]) > threshold:
        features_selected.append(index)

# Display of correlation matrix with selected features

df_new = df[features_selected]
corrmatrix_new = df_new.corr()
fig, ax = plt.subplots(figsize=(18, 10))
#sns.heatmap(corrmatrix_new, annot=True)  Uncomment to display the matrix

# Split the dataset

X = df_new.drop('stab', axis=1)
y = df_new['stab']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(f'Training set size: {X_train.shape}', f'   Test set size: {X_test.shape}')

# Standardization
# Standardization parameters are calculated on the training set and then applied to both the training and test set
scaler = StandardScaler()
scaler = scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

########################################################################################################################
#                                          Model Selection and Training                                                #
########################################################################################################################

models = []

Lasso_model = Lasso(alpha=1.0, fit_intercept=True, selection='random')
Lasso_parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
models.append('LASSO')

Ridge_model = Ridge(alpha=1.0, fit_intercept=True)
Ridge_parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
models.append('RIDGE')

MLP_model = MLPRegressor(hidden_layer_sizes=(200, ), activation='logistic', learning_rate_init=0.01, max_iter=300,
                         shuffle=True, early_stopping=True, beta_1=0.9, n_iter_no_change=20)
MLP_parameters = {'hidden_layer_sizes': [(50, 50, 50), (200, )], 'activation': ['relu', 'logistic',
                                                                                           'tanh']}
models.append('MLP')

#Hyperparameters Tuning and Best Model Selection

Lasso = GridSearchCV(Lasso_model, Lasso_parameters, scoring='r2', pre_dispatch='2*n_jobs', cv=10, refit=True,
                     verbose=0)
grid_lasso = Lasso.fit(X_train, y_train)
Lasso_model = grid_lasso.best_estimator_
print("Lasso best parameters : ", grid_lasso.best_params_)
print("Score of the Lasso model : ", Lasso_model.score(X_train, y_train))

Ridge = GridSearchCV(Ridge_model, Ridge_parameters, scoring='r2', pre_dispatch='2*n_jobs', cv=10, refit=True,
                     verbose=0)
grid_ridge = Ridge.fit(X_train, y_train)
Ridge_model = grid_ridge.best_estimator_
print("Ridge best parameters : ", grid_ridge.best_params_)
print("Score of the Ridge model : ", Ridge_model.score(X_train, y_train))

MLP = GridSearchCV(MLP_model, MLP_parameters, scoring='r2', pre_dispatch='2*n_jobs', cv=10, refit=True,
                     verbose=0)
grid_mlp = MLP.fit(X_train, y_train)
MLP_model = grid_mlp.best_estimator_
print("MLP best parameters : ", grid_mlp.best_params_)
print("Score of the MLP model : ", MLP_model.score(X_train, y_train))

########################################################################################################################
#                                              Test and Assessment                                                     #
########################################################################################################################

y_predicted = []

y_pred_lasso = Lasso_model.predict(X_test)
y_predicted.append(y_pred_lasso)

y_pred_ridge = Ridge_model.predict(X_test)
y_predicted.append(y_pred_ridge)

y_pred_mlp = MLP_model.predict(X_test)
y_predicted.append(y_pred_mlp)

for i in range(len(models)):
    print('/-------------------------------------------------------------------------------------------------------- /')
    print('                          Metrics of the %s Regressor' % models[i])
    print('/-------------------------------------------------------------------------------------------------------- /')
    print('RMSE is ', mean_squared_error(y_test, y_predicted[i], squared=False))
    print('MAE is ', mean_absolute_error(y_test, y_predicted[i]))
    print('R2 is ', r2_score(y_test, y_predicted[i]))

# Uncomment the following lines of code to display the residual plot

'''for j in range(len(models)):
    residuals = y_test - y_predicted[j]
    plt.figure()
    plt.scatter(x=y_predicted[j], y=residuals)
    plt.title('Residual plot: %s' % models[j])
    plt.grid(True)
    plt.xlabel("y_predicted")
    plt.ylabel("Residuals")

plt.show()'''









































