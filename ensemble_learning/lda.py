import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ensemble_learning.sklearn_utils import display_result, save_model, save_acc_result_txt

def lda_train(X_train, X_test, y_train, y_test):
    # define model
    model = LinearDiscriminantAnalysis()

    LDA = model.fit(X_train, y_train)

    train_labels = y_train['CreditLevel'].tolist()
    test_labels = y_test['CreditLevel'].tolist()

    train_predictions = LDA.predict(X_train)
    test_prediction = LDA.predict(X_test)

    # print("Train Result")
    # display_result(train_labels, train_predictions)
    print("Test Result")
    _, per_class_acc, __, ___ = display_result(test_labels, test_prediction)
    save_model(LDA, "lda.sav")
    save_acc_result_txt(per_class_acc, "lda.txt")

if __name__ == '__main__':
    df_ts = pd.read_csv('../data/BankChurners_normalized_standardized.csv')
    inputs = df_ts.iloc[0:,0:10]
    labels = df_ts.iloc[0:,10:11]

    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2,random_state=0)
    lda_train(X_train, X_test, y_train, y_test)