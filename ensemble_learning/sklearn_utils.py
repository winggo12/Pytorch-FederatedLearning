from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore")

def plot_cm(model_name, cm):
    for matrix in cm:
        fig = plt.figure()
        plt.matshow(cm)
        plt.title('Confusion Matrix of '+model_name)
        plt.colorbar()
        plt.ylabel('True Label')
        plt.xlabel('Predicated Label')
        plt.savefig('Confusion Matrix of ' +model_name+ '.jpg')

def display_result(labels, predictions,log=True):
    acc = accuracy_score(labels, predictions)
    cm = confusion_matrix(labels, predictions)
    cr = classification_report(labels, predictions)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    if log == True:
        print("------- result ------")
        print("Acc: ", acc)
        print("Per-Class Acc: ")
        print(per_class_acc)
        print("Report: ")
        print(cr)
        print("Acc: ", acc)
        print("Per-Class Acc: ")
    # print("Confusion Matrix: ")
    # print(cm)

    return acc, per_class_acc, cm, cr

def save_acc_result_txt(acc_list, filename):
    msg = ','.join(str(e) for e in acc_list)
    with open(filename, "w+") as f:
        f.write(msg)

def read_acc_result_txt(filename):
    acc = []
    with open(filename, "r") as filestream:
        for line in filestream:
            currentline = line.split(",")
            acc += currentline
    for i in range(len(acc)):
        acc[i] = float(acc[i])

    return acc

def save_model(model, filename):
    # save the model to disk
    # filename
    # = 'model.sav' for sklearn
    # = 'model.pkl' for xgboost classifier
    filename = filename
    pickle.dump(model, open(filename, 'wb'))

def load_model(model_path):
    # load the model from disk
    loaded_model = pickle.load(open(model_path, 'rb'))
    return loaded_model

def inference(model, X_test, y_test):

    test_labels = y_test['CreditLevel'].tolist()
    test_prediction = model.predict(X_test)

    return

if __name__ == '__main__':
    read_acc_result_txt('knn.txt')