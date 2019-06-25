from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingClassifier
from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
from sklearn.preprocessing import RobustScaler


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import xgboost as xgb

# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



def preprocess_data(data_path, labels_path=None):
    train_features = pd.read_csv(data_path,
                                 index_col=[0, 1, 2])

    features = ['reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k',
                'station_avg_temp_c',
                'station_min_temp_c']

    dataset = train_features[features]

    dataset = dataset.interpolate(method='linear')
    dataset = process_temp_features(dataset)

    # Trailing Moving Average
    # Tail-rolling average transform
    rolling = dataset.rolling(window=3, min_periods=1)
    rolling_mean = rolling.mean()

    #standarization
    transformer = RobustScaler().fit(rolling_mean)
    standarize_ds = pd.DataFrame(data=transformer.transform(rolling_mean)[0:, 0:],
                          index=dataset.index,
                          columns=features)

    if (labels_path):
        train_labels = pd.read_csv(labels_path,
                                   index_col=[0, 1, 2])
        standarize_ds = standarize_ds.join(train_labels)
    sj= standarize_ds.loc['sj']
    iq = standarize_ds.loc['iq']
    return sj, iq


def process_temp_features(dataset):
    for col_name in dataset.columns:
        if col_name.endswith('temp_k'):
            dataset[col_name] = dataset[col_name] - 273.15
    return dataset


def get_best_model(train, test):
    # Step 1: specify the form of the model
    model_formula = "total_cases ~ 1 + " \
                    "reanalysis_specific_humidity_g_per_kg + " \
                    "reanalysis_dew_point_temp_k + " \
                    "station_min_temp_c + " \
                    "station_avg_temp_c"

    grid = 10 ** np.arange(-8, -3, dtype=np.float64)

    best_alpha = []
    best_score = 1000

    # Step 2: Find the best hyper parameter, alpha
    for alpha in grid:
        model = smf.glm(formula=model_formula,
                        data=train,
                        family=sm.families.NegativeBinomial(alpha=alpha))

        results = model.fit()
        predictions = results.predict(test).astype(int)
        score = eval_measures.meanabs(predictions, test.total_cases)

        if score < best_score:
            best_alpha = alpha
            best_score = score

    print('best alpha = ', best_alpha)
    print('best score = ', best_score)

    # Step 3: refit on entire dataset
    full_dataset = pd.concat([train, test])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))

    fitted_model = model.fit()
    return fitted_model


def xg_boost_Algorithm_model(X_train, y_train, X_test, y_test):

    # gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)
    # y_pred = gbm.predict(X_test)

    # fit model no training data
    model = XGBClassifier()
    model.fit(X_train, y_train)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    return model

def gradient_boosting_algorithm(X_train, y_train, X_test, y_test):
    learning_rate = 0.1

    gb = GradientBoostingClassifier(loss="deviance", n_estimators=500, learning_rate= learning_rate, max_features="log2", max_depth=3,
                                    random_state=0)
    gb.fit(X_train, y_train)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_test, y_test)))
    return gb;


data_path = 'C:/Users/Dilshan/Documents/DengAI/data/dengue_features_train.csv'
label_path = 'C:/Users/Dilshan/Documents/DengAI/data/dengue_labels_train.csv'

sj_train, iq_train = preprocess_data(data_path, label_path)

sj_train_subtrain = sj_train.head(700).as_matrix()
sj_train_subtest = sj_train.tail(sj_train.shape[0] - 700).as_matrix()


sj_train_subtrain_X = sj_train_subtrain[:,0:4]
sj_train_subtrain_Y = sj_train_subtrain[:,4]

sj_train_subtest_X = sj_train_subtest[:,0:4]
sj_train_subtest_Y = sj_train_subtest[:,4]

sj_best_model = xg_boost_Algorithm_model(sj_train_subtrain_X, sj_train_subtrain_Y, sj_train_subtest_X, sj_train_subtest_Y)

sj_best_model_gradient = gradient_boosting_algorithm(sj_train_subtrain_X, sj_train_subtrain_Y, sj_train_subtest_X, sj_train_subtest_Y)

iq_train_subtrain = iq_train.head(400).as_matrix()
iq_train_subtest = iq_train.tail(iq_train.shape[0] - 400).as_matrix()


iq_train_subtrain_X = iq_train_subtrain[:,0:4]
iq_train_subtrain_Y = iq_train_subtrain[:,4]

iq_train_subtest_X = iq_train_subtest[:,0:4]
iq_train_subtest_Y = iq_train_subtest[:,4]


iq_best_model = xg_boost_Algorithm_model(iq_train_subtrain_X, iq_train_subtrain_Y, iq_train_subtest_X, iq_train_subtest_Y)
iq_best_model_gradient = gradient_boosting_algorithm(iq_train_subtrain_X, iq_train_subtrain_Y, iq_train_subtest_X, iq_train_subtest_Y)

sj_test, iq_test = preprocess_data('C:/Users/Dilshan/Documents/DengAI/data/dengue_features_test.csv')

sj_predictions = sj_best_model.predict(sj_test.as_matrix()).astype(int)
iq_predictions = iq_best_model.predict(iq_test.as_matrix()).astype(int)

sj_predictions_gradient = sj_best_model_gradient.predict(sj_test.as_matrix()).astype(int)
iq_predictions_gradient = iq_best_model_gradient.predict(iq_test.as_matrix()).astype(int)


submission = pd.read_csv("C:/Users/Dilshan/Documents/DengAI/data/submission_format.csv",
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
submission.to_csv("C:/Users/Dilshan/Documents/DengAI/data/benchmark_xg_boost_algorithm.csv")

submission.total_cases = np.concatenate([sj_predictions_gradient, iq_predictions_gradient])
submission.to_csv("C:/Users/Dilshan/Documents/DengAI/data/benchmark_gradient_boosting_algorithm.csv")
print("Done")