import matplotlib
import pandas as pd
import numpy as np
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from sklearn import tree
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn import model_selection, svm, metrics
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from matplotlib import font_manager
import lightgbm as lgb
import sys
from sklearn.metrics import accuracy_score

from apps.mtransfer.transfer.transfer_data.TrAdaboost import TradaBoostClassifier

sys.path.append(r'..')

mfont = font_manager.FontProperties(fname='C:\Windows\Fonts\simfang.ttf')


def append_feature(dataframe):
    X = dataframe.values
    X = X[:, 1:X.shape[1] - 1]
    total_S = np.sum(X, axis=1)
    var_S = np.var(X, axis=1)
    X = np.c_[X, total_S]
    X = np.c_[X, var_S]

    return X
def train_compare(num,algo):
    global lgb

    SEED = 2
    np.random.seed(SEED)
    train_df = pd.read_table('apps/mtransfer/transfer/transfer_data/A_train.csv', encoding='utf-8', sep='\t', engine='python', index_col=0)
    train_df1 = pd.read_table('apps/mtransfer/transfer/transfer_data/B_train.csv', encoding='utf-8', sep='\t', engine='python', index_col=0)
    test_df = pd.read_table("apps/mtransfer/transfer/transfer_data/B_test.csv", encoding="utf-8", sep='\t', engine='python', index_col=0)
    train_df1 = pd.concat([train_df1,test_df])
    # train_df1 = train_df1.sample(n=100,random_statu=SEED)
    train_df = train_df[train_df['do_flag'] != 3]
    train_df1 = train_df1[train_df1['do_flag'] != 3]
    test_df = test_df[test_df['do_flag'] != 3]
    train_df.replace(2, 0, inplace=True)
    train_df1.replace(2, 0, inplace=True)
    test_df.replace(2, 0, inplace=True)
    train_data_T = train_df.values
    train_data_S = train_df1.values
    test_data_S = test_df.values
    print('data loaded.')
    label_T = train_data_T[:, train_data_T.shape[1] - 1]
    trans_T = append_feature(train_df)
    label_S = train_data_S[:, train_data_S.shape[1] - 1]
    trans_S = append_feature(train_df1)

    def train_lgbm(train_data, label_data, test_data, test_label):
        lgbm = lgb.LGBMClassifier(learning_rate=0.01, max_depth=3, bagging_fraction=0.6, feature_fraction=0.6,
                                  num_leaves=8, bagging_freq=5, cat_smooth=1, lambda_l2=0)
        lgbm.fit(train_data, label_data)
        pred = lgbm.predict_proba(test_data)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(y_true=test_label, y_score=pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        return auc

    def train_tradaboost(trans_S, trans_T, label_S, label_T, test_data_S, test_label, N, clf):
        trc = TradaBoostClassifier(epoch=N, learner=clf)
        trc.fit(trans_S, trans_T, label_S, label_T, test_data_S)
        pred = trc.predict_proba(test_data_S)
        fpr, tpr, thresholds = metrics.roc_curve(y_true=test_label, y_score=pred[0], pos_label=1)
        tauc = metrics.auc(fpr, tpr)
        return tauc

    def train_adaboost(train_data, label_data, test_data, test_label):
        bdt = AdaBoostClassifier(tree.DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=5),
                                 algorithm='SAMME.R', learning_rate=0.1, n_estimators=50)
        bdt.fit(train_data, label_data)
        pred = bdt.predict_proba(test_data)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(y_true=test_label, y_score=pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        return auc

    tlgb = lgb.LGBMClassifier()
    x1 = (0.2*num) / 100
    print(num)
    print("x1",x1)
    X_train2, X_test2, y_train2, y_test2 = model_selection.train_test_split(trans_S, label_S, test_size = x1,random_state=7)
    print('源数据集数量', X_test2.shape)
    if algo == 1:
        auc_tradaboost = train_tradaboost(X_test2, trans_T, y_test2, label_T, X_train2, y_train2, 8, tlgb)
        return round(auc_tradaboost,5)
    elif algo == 2:
        auc_lgb = train_lgbm(X_test2, y_test2, X_train2, y_train2)
        return round(auc_lgb,5)
    else:
        auc_ada = train_adaboost(X_test2, y_test2, X_train2, y_train2)
        return round(auc_ada,5)