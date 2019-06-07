from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
from sklearn import preprocessing


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

    # scaling the dataset
    mm_scaler = preprocessing.MinMaxScaler()
    _train_minmax = mm_scaler.fit_transform(dataset)

    scaled_ds = pd.DataFrame(data=_train_minmax[0:, 0:],
                          index=dataset.index,
                          columns=features)

    if (labels_path):
        train_labels = pd.read_csv(labels_path,
                                   index_col=[0, 1, 2])
        scaled_ds = scaled_ds.join(train_labels)
    sj= scaled_ds.loc['sj']
    iq = scaled_ds.loc['iq']
    return sj, iq

def process_temp_features(dataset):
    for col_name in dataset.columns:
        if col_name.endswith('temp_k'):
            dataset[col_name] = dataset[col_name] - 273.15
    return dataset


data_path = '/home/karu/PycharmProjects/DengAI/data/dengue_features_train.csv'
label_path = '/home/karu/PycharmProjects/DengAI/data/dengue_labels_train.csv'

sj_train, iq_train = preprocess_data(data_path, label_path)


# def fix_col_outliers(column, fence_low, fence_high, win_size=5):
#
#
# def clean_training_data(dataset):
#     for col_name in dataset.columns.values:
#         q1 = dataset[col_name].quantile(0.25)
#         q3 = dataset[col_name].quantile(0.75)
#         iqr = q3 - q1  # Interquartile range
#         fence_low = q1 - 1.5 * iqr
#         fence_high = q3 + 1.5 * iqr
#         dataset[col_name] = fix_col_outliers(dataset[col_name], fence_low, fence_high)
#     return dataset

# sj_train, iq_train = clean_train_data(sj_train), clean_train_data(iq_train)

print(sj_train.shape)
print(iq_train.shape)

sj_train_subtrain = sj_train.head(700)
sj_train_subtest = sj_train.tail(sj_train.shape[0] - 700)

iq_train_subtrain = iq_train.head(400)
iq_train_subtest = iq_train.tail(iq_train.shape[0] - 400)


def get_best_model(train, test):
    # Step 1: specify the form of the model
    model_formula = "total_cases ~ 1 + " \
                    "reanalysis_specific_humidity_g_per_kg + " \
                    "reanalysis_dew_point_temp_k + " \
                    "station_min_temp_c + " \
                    "station_avg_temp_c"

    grid = 10 ** np.arange(-8, -3, dtype=np.float64)

    print(len(grid))

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


sj_best_model = get_best_model(sj_train_subtrain, sj_train_subtest)
iq_best_model = get_best_model(iq_train_subtrain, iq_train_subtest)


sj_test, iq_test = preprocess_data('/home/karu/PycharmProjects/DengAI/data/dengue_features_test.csv')

sj_predictions = sj_best_model.predict(sj_test).astype(int)
iq_predictions = iq_best_model.predict(iq_test).astype(int)

submission = pd.read_csv("/home/karu/PycharmProjects/DengAI/data/submission_format.csv",
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
submission.to_csv("/home/karu/PycharmProjects/DengAI/data/benchmark.csv")
