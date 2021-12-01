import numpy as np
import pandas as pd
import torch
from ensemble_learning.sklearn_utils import load_model , read_acc_result_txt, display_result, plot_cm
from dataset_processing.dataset import TheDataset, TheDatasetByDataframe
from dataset_preprocessing.dataset_preprocess import dataset_preprocess
from inference import inference_nn_by_df

file_path = "../data/New_BankChurners.csv"
X_test = dataset_preprocess(file_path)
X_test.drop(["CreditLevel"], axis=1, inplace=True)

# y_test = pd.read_csv('../data/y_test.csv')
# test_labels = y_test['CreditLevel'].tolist()

row_num, col_num = X_test.shape[1], X_test.shape[0]

acc_result_dict = {}

#Running the saved model of XGB Classifier
xgb = load_model(model_path='xgbc.pkl')
xgb_test_acc = read_acc_result_txt('xgbc.txt')
xgb_result = xgb.predict(X_test)
acc_result_dict['xgb'] = [xgb_test_acc, xgb_result]

#Running the saved model of Decision Tree
dt = load_model(model_path='dt.sav')
dt_test_acc = read_acc_result_txt('dt.txt')
dt_result = xgb.predict(X_test)
acc_result_dict['dt'] = [dt_test_acc, dt_result]

#Running the saved model of KNN
knn = load_model(model_path='knn.sav')
knn_test_acc = read_acc_result_txt('knn.txt')
knn_result = knn.predict(X_test)
acc_result_dict['knn'] = [knn_test_acc, knn_result]

#Running the saved model of LDA
lda = load_model(model_path='lda.sav')
lda_test_acc = read_acc_result_txt('lda.txt')
for i in range(len(lda_test_acc)):
    lda_test_acc[i] = lda_test_acc[i]*0.2
lda_result = lda.predict(X_test)
acc_result_dict['lda'] = [lda_test_acc, lda_result]

#Running the saved model of NN
nn_result = inference_nn_by_df(model_path="./model.pth", input_df=X_test)
nn_test_acc = read_acc_result_txt('model.txt')
for i in range(len(nn_test_acc)):
    nn_test_acc[i] = nn_test_acc[i]*0.2
acc_result_dict['nn'] = [nn_test_acc, nn_result]

#Start Ensemble Learning by Combining All Result
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

ensemblelearning_result = []
for score in final_scores:
    index = list(score).index(max(score))
    label = index + 1
    ensemblelearning_result.append(label)

print(ensemblelearning_result)

df = pd.read_csv(file_path)
df.drop(["CreditLevel"], axis=1, inplace=True)
df['CreditLevel'] = ensemblelearning_result

df.to_csv('../data/your_student_ID_bonus.csv', index=False)


print("End")

