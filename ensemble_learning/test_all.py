import numpy as np
import pandas as pd
import torch
from ensemble_learning.sklearn_utils import load_model , read_acc_result_txt, display_result, plot_cm
from dataset_processing.dataset import TheDataset, TheDatasetByDataframe
from inference import inference_nn_by_df

X_train = pd.read_csv('../data/X_train.csv')
X_test = pd.read_csv('../data/X_test.csv')
y_train = pd.read_csv('../data/y_train.csv')
y_test = pd.read_csv('../data/y_test.csv')

test_labels = y_test['CreditLevel'].tolist()

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
nn_result, nn_labels = inference_nn_by_df(model_path="./model.pth", input_df=X_test, label_df=y_test)
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

##Display the result of all ML methods and EL final results
print("XGB Classifier")
display_result(test_labels, xgb_result)
print("Decision Tree")
display_result(test_labels, dt_result)
print("K-Nearest Neighbor")
display_result(test_labels, knn_result)
print("LDA")
display_result(test_labels, lda_result)
print("Neural Network")
_,_,cm,_ = display_result(test_labels, nn_result)
plot_cm(model_name="FcNet", cm= cm) ##confusion matrix
print("Ensemble Learning")
_,_,cm,_=display_result(test_labels, ensemblelearning_result)
plot_cm(model_name="Ensemble Learning", cm= cm) ##confusion matrix



