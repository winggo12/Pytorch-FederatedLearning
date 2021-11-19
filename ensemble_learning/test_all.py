import numpy as np
import pandas as pd
import torch
from ensemble_learning.sklearn_utils import load_model , read_acc_result_txt, display_result
from datasetProcessing.dataset import TheDataset, TheDatasetByDataframe
from inference import inference_nn

X_train = pd.read_csv('../data/X_train.csv')
X_test = pd.read_csv('../data/X_test.csv')
y_train = pd.read_csv('../data/y_train.csv')
y_test = pd.read_csv('../data/y_test.csv')

test_labels = y_test['CreditLevel'].tolist()

row_num, col_num = X_test.shape[1], X_test.shape[0]

acc_result_dict = {}

xgb = load_model(model_path='xgbc.pkl')
xgb_test_acc = read_acc_result_txt('xgbc.txt')
xgb_result = xgb.predict(X_test)
acc_result_dict['xgb'] = [xgb_test_acc, xgb_result]

dt = load_model(model_path='dt.sav')
dt_test_acc = read_acc_result_txt('dt.txt')
dt_result = xgb.predict(X_test)
acc_result_dict['dt'] = [dt_test_acc, dt_result]

knn = load_model(model_path='knn.sav')
knn_test_acc = read_acc_result_txt('knn.txt')
knn_result = knn.predict(X_test)
acc_result_dict['knn'] = [knn_test_acc, knn_result]
display_result(test_labels, knn_result)

lda = load_model(model_path='lda.sav')
lda_test_acc = read_acc_result_txt('lda.txt')
for i in range(len(lda_test_acc)):
    lda_test_acc[i] = lda_test_acc[i]*0.2
lda_result = lda.predict(X_test)
acc_result_dict['lda'] = [lda_test_acc, lda_result]
display_result(test_labels, lda_result)

bank_dataset = TheDatasetByDataframe(input_df=X_test, label_df=y_test)
data_size = bank_dataset.X_.shape[0]
data_loader = torch.utils.data.DataLoader(bank_dataset, batch_size=data_size, shuffle=True)
nn_result, ground_truth = inference_nn(model_path="./model.pth", data_loader=data_loader)
nn_test_acc = read_acc_result_txt('model.txt')
for i in range(len(nn_test_acc)):
    nn_test_acc[i] = nn_test_acc[i]*0.2

acc_result_dict['nn'] = [nn_test_acc, nn_result]
display_result(test_labels, nn_result)

zeros = np.zeros(shape=(10))

final_scores = []

for i in range(col_num):
    tmp = zeros.copy()
    for k, v in acc_result_dict.items():
        acc , result = v[0], v[1]
        label = result[i]
        index = label - 1
        tmp[index] += acc[index]
        # tmp[index] += 1
    final_scores.append(tmp)

test_prediction = []
for score in final_scores:
    index = list(score).index(max(score))
    label = index + 1
    test_prediction.append(label)

_, per_class_acc, __, ___ = display_result(test_labels, test_prediction)

