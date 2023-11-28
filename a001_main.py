import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from a002_net import MyModel, weight_init

ALL_DATA_PATH = '../all_data.csv'
IMPORTANT_COLS_TXT = "./important_cols.txt"
FILTERED_DATA_PATH = './filtered_data.csv'
TENSOR_TYPE = torch.float32
DEVICE = torch.device('cuda')
K = 5
LR = 0.1
BATCH_SIZE = 4096
TOTAL_EPOCHS = 10
NUM_WORKERS = 8

LOSS_FUNC = nn.CrossEntropyLoss()


def start():
    all_df = pd.read_csv(ALL_DATA_PATH)
    print(all_df.dtypes)
    label_count = all_df['Label'].value_counts()  # 返回类型Series
    print(label_count)

    grouped_df_dic = {key: df for key, df in all_df.groupby('Label')}
    for key, df in grouped_df_dic.items():
        df.to_csv(f"./classified_data/{key}.csv", index=False)


def preprocess_data_and_save():
    dic = get_important_column_dict()
    selected_col_name_list = []
    for key, value in dic.items():
        selected_col_name_list += value
    print(selected_col_name_list)

    all_data_df = pd.read_csv(ALL_DATA_PATH)
    selected_col_name_list = list(set(selected_col_name_list))
    filtered_df = all_data_df[selected_col_name_list]  # 筛掉了不需要的列

    # 删掉一些类别很少的行
    label_series = all_data_df['Label']
    row_index_to_remove = []
    for index, value in enumerate(label_series):
        if value == ("Infiltration" or "Heartbleed"):
            row_index_to_remove.append(index)
    filtered_df = filtered_df.drop(row_index_to_remove)
    label_series.drop(row_index_to_remove, inplace=True)

    filtered_df = filtered_df.astype('float32')
    filtered_df = filtered_df.apply(lambda x: (x - x.mean()) / (x.std() + 1e-10))
    filtered_df.fillna(0)

    filtered_df['Label'] = label_series
    filtered_df.to_csv(FILTERED_DATA_PATH, index=False)


def read_prepared_data(only_pred_two_classes=False):
    """Prior: preprocess_data_and_save"""
    df = pd.read_csv(FILTERED_DATA_PATH)
    labels = df['Label']
    del df['Label']

    dic = {}
    for int_index, (text_index, value) in enumerate(labels.value_counts().items()):
        # 如果只分两类，则正常流量为0，攻击为1；否则攻击流量要细分类。
        if only_pred_two_classes:
            if int_index == 0:
                dic[text_index] = int_index
            else:
                dic[text_index] = 1
        else:
            dic[text_index] = int_index

    # for i in range(len(labels)):
    #     labels[i] = dic[labels[i]]
    labels = labels.apply(lambda x: dic[x])  # 不要在这里用显示的循环。

    training_set = df.to_numpy()
    labels = labels.to_numpy()
    return training_set, labels


def ndarray_to_tensor(a, dtype=TENSOR_TYPE):
    return torch.tensor(data=a,
                        dtype=dtype)


def train_mlp():
    total_features, total_labels = read_prepared_data(only_pred_two_classes=True)
    net = MyModel().to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    k_fold = KFold(n_splits=K, shuffle=True)
    current_k = 0
    for train_index, test_index in k_fold.split(total_labels):
        x_train, x_vali = total_features[train_index], total_features[test_index]
        y_train, y_vali = total_labels[train_index], total_labels[test_index]

        x_train, x_vali = ndarray_to_tensor(x_train), ndarray_to_tensor(x_vali)
        y_train, y_vali = ndarray_to_tensor(y_train, torch.int64), ndarray_to_tensor(y_vali, torch.int64)

        training_features_labels = TensorDataset(x_train, y_train)
        vali_features_labels = TensorDataset(x_vali, y_vali)
        training_dtl = DataLoader(training_features_labels,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS)
        vali_dtl = DataLoader(vali_features_labels,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=NUM_WORKERS)
        train_and_vali(training_dtl, vali_dtl, net, optimizer, current_k)
        current_k += 1


def train_ml(chosen_model="forest"):
    """
    Use a model to train, and show confusion matrix results on validation dataset.
    Args:
        chosen_model: str in ["forest, adaboost, bayes"]
    """
    if chosen_model == "forest":
        ml_model = RandomForestClassifier(max_depth=20,
                                          n_estimators=50,
                                          n_jobs=-1)
    elif chosen_model == "adaboost":
        # base_model = DecisionTreeClassifier(max_depth=10)
        ml_model = AdaBoostClassifier()
    elif chosen_model == "bayes":
        ml_model = GaussianNB()
    else:
        print("No such model.")
        exit()

    total_features, total_labels = read_prepared_data(only_pred_two_classes=True)
    k_fold = KFold(n_splits=K, shuffle=True)
    current_k = 0
    for train_index, test_index in k_fold.split(total_labels):
        x_train, x_vali = total_features[train_index], total_features[test_index]
        y_train, y_vali = total_labels[train_index], total_labels[test_index]
        # 训练
        ml_model.fit(x_train, y_train)
        # 验证
        y_hat = ml_model.predict(x_vali)

        accuracy = accuracy_score(y_vali, y_hat)
        print(f"k = {current_k}, accuracy = {accuracy * 100}")

        show_confusion_matrix(y_hat, y_vali)

        current_k += 1

    # print(forest.feature_importances_)


def train_and_vali(training_dtl, vali_dtl, net, optimizer, current_k):
    net.apply(weight_init)
    for epoch in range(TOTAL_EPOCHS):
        net.train()
        iter_cnt = 0
        for x, y in training_dtl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_hat = net(x)
            loss = LOSS_FUNC(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"k = {current_k}, epoch = {epoch}, iter = {iter_cnt}, loss = {loss}")
            iter_cnt += 1

        with torch.no_grad():
            net.eval()
            y_pred_list = []
            y_list = []
            for x, y in vali_dtl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_hat = net(x)
                y_hat = torch.argmax(y_hat, dim=1)
                y_pred_list += y_hat.detach().cpu().tolist()
                y_list += y.detach().cpu().tolist()
            y_pred_tensor = torch.tensor(y_pred_list)
            y_tensor = torch.tensor(y_list)

            show_confusion_matrix(y_pred_tensor, y_tensor)


def show_confusion_matrix(y_pred, y):
    ConfusionMatrixDisplay.from_predictions(y, y_pred)
    plt.show()


def get_important_column_dict():
    """返回字典"""
    with open(IMPORTANT_COLS_TXT, 'r', encoding='utf-8') as txt:
        lines = txt.readlines()
        txt.close()

    dic = {}
    for line in lines:
        tokens = remove_some_chars(line).split('=')
        key = tokens[0]
        col_names_list = tokens[1].split(',')
        col_names_list = [name.strip() for name in col_names_list]

        dic[key] = col_names_list
    return dic


def remove_some_chars(string: str):
    chars_need_to_be_removed = ['[', ']', '"', '\n']
    for char in chars_need_to_be_removed:
        string = string.replace(char, '')
    return string


if __name__ == '__main__':
    # start()
    # preprocess_data_and_save()
    # train_mlp()
    train_ml(chosen_model="forest")
