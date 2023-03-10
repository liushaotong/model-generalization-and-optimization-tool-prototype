import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from Setting import *
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def kendall_tau(data_train):
    kt_list = []
    for i in range(32):
        tau_gen_gap, p_value = stats.kendalltau(data_train.loc[:, ['gen.gap']], data_train.iloc[:, i])
        print(data_train.columns[i], '\t\t', tau_gen_gap)
        if tau_gen_gap < 0.4:
            kt_list.append((data_train.columns[i]))

    return kt_list


def to_excel(data_train, kt_list=None):
    if kt_list != None:
        for name in kt_list:
            data_train = data_train.drop([name], axis=1)
    else:
        kt_list = []
    headers = data_train.columns.values[:32-len(kt_list)]
    df = pd.DataFrame(columns=headers)
    for i in range(32-len(kt_list)):
    # for i in range(32):
        df.iloc[:, i] = data_train.iloc[:, i]
    df.to_csv('./svhn_unfilter.csv')
    return


def plot_one(pre_y_list, y):
    pre_y_list = pre_y_list * 100
    y = y * 100
    plt.plot(pre_y_list, y, 'o', markersize=1.0)  # 画出每条预测结果线

    plt.title('regression result comparison on SVHN')  # 标题
    plt.legend(loc='upper right')  # 图例位置
    plt.ylabel('real value')  # y轴标题
    plt.xlabel('predicted value')
    plt.savefig('true_pred_svhn.jpg')  # 保存图片
    plt.show()  # 展示图像
    return


def plot_two(pre_y_list, y):
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    list_len = len(pre_y_list)
    X = range(list_len)
    s1 = np.pi * 2 ** 2
    # plt.scatter(X, y, s1, 'g', marker='.', alpha=0.4)
    # plt.xlabel("index")
    # plt.ylabel("gen gap")
    # plt.title("result")
    # data = np.random.randn(10000)

    weights = np.ones_like(y) / float(len(y))
    plt.hist(y, bins=10, weights=weights, facecolor="blue", edgecolor="black", alpha=0.7)
    # 显示横轴标签
    plt.xlabel("区间")
    # 显示纵轴标签
    plt.ylabel("频率")
    # 显示图标题
    plt.title("频率分布直方图")
    plt.savefig('cifar10_model_distribution.jpg')
    plt.show()
    return


def main():
    data_train = pd.read_csv('./nin.cifar10_svhn.csv')
    # data_train['hp.dataset']=='cifar10' or 'svhn  #超参数设置
    data_train = data_train[np.logical_and(data_train['hp.dataset']==task, data_train['is.converged']==True, data_train['is.high_train_accuracy']==True)]
    # kt_list = kendall_tau(data_train)
    # kt_list = None
    # to_excel(data_train, kt_list)


    # data_train = pd.read_csv('./test1.csv')
    # print(data_train.info())

    data_train_target = data_train.loc[:, ['gen.gap']]
    y = data_train_target.values.flatten()
    # data_train_x = pd.read_csv('./svhn_unfilter.csv')  # 超参数设置
    # X = data_train_x.values[:, 1:]
    X = data_train[train_list].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    # xgboost、SVR、RFR回归
    # regr = XGBRegressor(max_depth=3, n_estimators=500, random_state=400, learning_rate=0.1, gamma=0.2,
    #                     min_child_weight=6,
    #                     reg_alpha=1, reg_lambda=1)
    # regr = SVR()
    regr = RandomForestRegressor()
    # regr = GradientBoostingRegressor()

    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    pre_y_list = regr.predict(X)


    print("训练集合上RMSE = {:.3f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
    print("训练集合上R^2 = {:.3f}".format(r2_score(y_test, y_pred)))
    tau_gen_gap, p_value = stats.kendalltau(y_test, y_pred)
    print(tau_gen_gap)
    return


if __name__ == '__main__':
    main()
