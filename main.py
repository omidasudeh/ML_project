import pandas as pd
import numpy as np
import math
import copy as cp
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer


categorical_attrib_values = { "workclass":['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],                 "education":['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],                 "marital.status":['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],                 "occupation":['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],                 "relationship":['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],                 "race":['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],                 "sex":['Female', 'Male'],                 "native.country": ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
                }

train_df = pd.read_csv("./data/train_final.csv",                        dtype={'workclass':'category',                              'education':'category',                              'marital.status':'category',                              'occupation':'category',                              'relationship':'category',                              'race':'category',                              'sex':'category',                              'native.country':'category'
                             })
m_train = len(train_df)
test_df = pd.read_csv("./data/test_final.csv",                      dtype={'workclass':'category',                              'education':'category',                              'marital.status':'category',                              'occupation':'category',                              'relationship':'category',                              'race':'category',                              'sex':'category',                              'native.country':'category'
                             })
m_test = len(test_df)


# # preprocess the data
# - workclass, occupation, native.country has anomaly data '?'

def convert_numerical_to_binary(df):
    for col_name in df.columns:
        if col_name!= "ID" and col_name not in categorical_attrib_values:
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            threashold = df[col_name].median()
            print("preprocessing numerical attribute:", col_name, "threashold:", threashold)
            df[col_name] = df[col_name].apply(lambda x: 0 if int(x) <= threashold else 1)

train_df_processed = train_df
convert_numerical_to_binary(train_df_processed)

train_mode_country = train_df_processed["native.country"].mode()[0]
train_mode_workclass = train_df_processed["workclass"].mode()[0]
train_mode_occupation = train_df_processed["occupation"].mode()[0]
train_df_processed["native.country"]= train_df_processed["native.country"].replace("Holand-Netherlands",train_mode_country, regex=True)
train_df_processed["native.country"] = train_df_processed["native.country"].apply(lambda x: train_mode_country if x=='?' else x)
train_df_processed["workclass"] = train_df_processed["workclass"].apply(lambda x: train_mode_workclass if x=='?' else x)
train_df_processed["occupation"] = train_df_processed["occupation"].apply(lambda x: train_mode_occupation if x=='?' else x)


test_df_processed = test_df
convert_numerical_to_binary(test_df_processed)

test_mode_country = test_df_processed["native.country"].mode()[0]
test_mode_workclass = test_df_processed["workclass"].mode()[0]
test_mode_occupation = test_df_processed["occupation"].mode()[0]
test_df_processed["native.country"]= test_df_processed["native.country"].replace("Holand-Netherlands",test_mode_country, regex=True)
test_df_processed["native.country"] = test_df_processed["native.country"].apply(lambda x: test_mode_country if x=='?' else x)
test_df_processed["workclass"] = test_df_processed["workclass"].apply(lambda x: test_mode_workclass if x=='?' else x)
test_df_processed["occupation"] = test_df_processed["occupation"].apply(lambda x: test_mode_occupation if x=='?' else x)

## Splitting fetures and labels
features = list(train_df.columns[:14])
y = train_df["income>50K"]
X = pd.get_dummies(train_df_processed[features],drop_first=True)
test_features = list(test_df.columns[1:])
test_X = pd.get_dummies(test_df_processed[test_features],drop_first=True)


# # Fit a tree

dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X, y)
preds = dt.predict(X)
incorrect = np.count_nonzero(preds-y)
print("train_accuracy", 1 - incorrect/X.shape[0])

test_df_processed["Prediction"] = dt.predict(test_X)
result = test_df_processed[["ID","Prediction"]]
result.to_csv("sl-dt.csv", index=False)


# # Regression Tree

dtr = DecisionTreeRegressor() 
dtr.fit(X, y)
test_df_processed["Prediction"] = dtr.predict(test_X)
result = test_df_processed[["ID","Prediction"]]
result.to_csv("sl-dtr.csv", index=False)


# # Random Forest

regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X, y)
test_df_processed["Prediction"] = regr.predict(test_X)
result = test_df_processed[["ID","Prediction"]]
result.to_csv("sl-rf.csv", index=False)


# # GradientBoosting Best

reg = GradientBoostingRegressor(n_estimators=200, learning_rate=0.2387755102040816, max_depth=5, random_state=0,loss='squared_error')
reg.fit(X, y)
test_df_processed["Prediction"] = reg.predict(test_X)
result = test_df_processed[["ID","Prediction"]]
result.to_csv("sl-eb.csv", index=False)


# # Neural Network

regr = MLPRegressor(random_state=1, max_iter=500,hidden_layer_sizes= (25,11,7,5,3,) ).fit(X, y)
test_df_processed["Prediction"] = regr.predict(test_X)
result = test_df_processed[["ID","Prediction"]]
result.to_csv("sl-MLP.csv", index=False)


# # Gradient boosting with cross validation

# Number of trees to be used
xgb_n_estimators = [int(x) for x in np.linspace(20,1000, 15)]

# Maximum number of levels in tree
xgb_max_depth = [int(x) for x in np.linspace(2, 20, 5)]

# Minimum number of instaces needed in each node
xgb_min_child_weight = [int(x) for x in np.linspace(1, 10, 10)]

# Tree construction algorithm used in XGBoost
xgb_tree_method = ['auto', 'exact', 'approx', 'hist', 'gpu_hist']

# Learning rate
xgb_eta = [x for x in np.linspace(0.01, 0.2, 5)]

# Minimum loss reduction required to make further partition
xgb_gamma = [int(x) for x in np.linspace(0, 0.5, 5)]

# Learning objective used
xgb_objective = ['reg:squarederror']#, 'reg:squaredlogerror']
# Create the grid
xgb_grid = {'n_estimators': xgb_n_estimators,
            'max_depth': xgb_max_depth,
            'min_child_weight': xgb_min_child_weight,
            'tree_method': xgb_tree_method,
            'eta': xgb_eta,
            'gamma': xgb_gamma,
            'objective': xgb_objective}
# Create the model to be tuned
xgb_base = xgb.XGBRegressor()
# Create the random search Random Forest
xgb_random = RandomizedSearchCV(estimator = xgb_base, param_distributions = xgb_grid, 
                                n_iter = 30, cv = 3, verbose = 2, 
                                random_state = 1, n_jobs = -1)
# Fit the random search model
xgb_random.fit(X, y)

# # Get the optimal parameters
print(xgb_random.best_params_)

grid = {'objective': 'reg:squarederror', 'n_estimators': 860, 'max_depth': 6, 'gamma': 0, 'eta': 0.01}
xgb_base = xgb.XGBRegressor(**grid)
xgb_base.fit(X, y)
test_df_processed["Prediction"] = xgb_base.predict(test_X)
result = test_df_processed[["ID","Prediction"]]
result.to_csv("sl-xgb-approx.csv", index=False)

