import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')
import lightgbm as lgb
import pickle

train = pd.read_csv('train_sample.csv', header=0)
test = pd.read_csv('test.csv', header=0)
test.index = test.click_id
test.drop('click_id', axis=1, inplace=True)


def data_prep(data, train=pd.DataFrame(index=range(1, 10))):
    # convert categorical data from intergers to categories
    Feature = ['ip', 'app', 'device', 'os', 'channel']
    for feature in Feature:
        data[feature] = data[feature].astype('category')

    # set click time to time series
    data['click_time'] = pd.to_datetime(data['click_time'])
    data['click_hour'] = data['click_time'].dt.hour

    # check for minute patterns
    data['click_min'] = data['click_time'].dt.minute

    # Adding a confidence rates for is_attributed
    ATTRIBUTION_CATEGORIES = [
        ['ip'], ['app'], ['device'], ['os'], ['channel'],
        ['app', 'channel'],
        ['app', 'os'],
        ['app', 'device'],
        ['channel', 'os'],
        ['channel', 'device'],
        ['os', 'device']
    ]

    freqs = {}
    try:
        for cols in ATTRIBUTION_CATEGORIES:
            new_feature = '_'.join(cols) + '_confRate'
            group_object = data.groupby(cols)
            group_sizes = group_object.size()
            log_group = np.log(100000)

            def rate_calculation(x):
                rate = x.sum() / float(x.count())
                conf = np.min([1, np.log(x.count()) / log_group])
                return rate * conf

            data = data.merge(
                group_object['is_attributed'].apply(rate_calculation).reset_index().rename(
                    index=str,
                    columns={'is_attributed': new_feature}
                )[cols + [new_feature]],
                on=cols, how='left')
    except:
        for cols in ATTRIBUTION_CATEGORIES:
            new_feature = '_'.join(cols) + '_confRate'
            temp = train[cols + [new_feature]]
            temp.drop_duplicates(inplace=True)
            data = pd.merge(data, temp, on=cols, how='left')

    data.drop('click_time', axis=1, inplace=True)
    try:
        data.drop('attributed_time', axis=1, inplace=True)
    except:
        pass
    return data


train = data_prep(train)
test = data_prep(test, train=train)

Target = ['is_attributed']
Feature = test.columns.tolist()
try:
    Feature.remove('is_attributed')  # ['ip', 'app', 'device', 'os', 'channel']
except:
    pass

# train,test = train_test_split(data, train_size = 0.7, test_size = 0.3, random_state = 123)
# x, y = train[Feature].values, train[Target].values.ravel()
x, y = train[Feature].values, train[Target].values.ravel()
train_data = lgb.Dataset(x, label=y)
# test_data = lgb.Dataset(test[Feature].values, label = test[Target].values)

param = {
    'application': 'binary',
    'objective': 'binary',
    'metirc': 'auc',
    #     'is_unbalance':'true',
    'scale_pos_weight': 50,
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}
clf_lgb = lgb.train(param, train_data)
pred_prob = clf_lgb.predict(test[Feature].values)
test['is_attributed'] = pred_prob

# pred_01 = np.where(pred_prob>0.5,1,0)
# print(roc_auc_score(test[Target].values, pred_prob))

# Save model
filename = 'Lightgbm_TalkingData.sav'
pickle.dump(clf_lgb, open(filename, 'wb'))

test['click_id'] = test.index
test.to_csv('talking data.csv',sep = ',', header = True, columns=['click_id','is_attributed'], index = False)