import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score,confusion_matrix,f1_score, recall_score,precision_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb

# READ DATA
data = pd.read_csv('train.csv',header = 0)
Target = ['is_attributed']
Feature = ['ip', 'app', 'device', 'os', 'channel']

train,test = train_test_split(data, train_size = 0.9, test_size = 0.1, random_state = 123)
x, y = train[Feature].values, train[Target].values.ravel()
train_data = lgb.Dataset(x,label = y)
test_data = lgb.Dataset(test[Feature].values, label = test[Target].values)

param = {
    'application':'binary',
    'objective':'binary',
    'metirc':'auc',
#     'is_unbalance':'true',
    'scale_pos_weight': 50,
    'boosting':'gbdt',
    'num_leaves':31,
    'feature_fraction':0.5,
    'bagging_freq':20,
    'learning_rate':0.05,
    'verbose':0
}

clf_lgb = lgb.train(param, train_data)
pred_prob = clf_lgb.predict(test[Feature].values)
pred_01 = np.where(pred_prob>0.5,1,0)
print(roc_auc_score(test[Target].values, pred_prob))