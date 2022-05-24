# Thijs Ermens
# 10-5-2022
# Dit script kan met het parquet file voorspellen of het gen benign of
# pathogeen is

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import math


# Met Conservation_alt en Concervation_ref
# De accuray is:  0.8432096723811624
# De MCC is:  0.37277089471451313
# De FDR is:  0.05749755707559741

# Met Conservation_alt en Blosumscore
# De accuray is:  0.8266725465336346
# De MCC is:  0.315280234067626
# De FDR is:  0.06976433673810001

def k_nearest_neighbors(train, val, test):
    df = pd.read_csv(train)
    X = df[['Concervation_alt', 'Concervation_ref']].values
    y = df['Pathogeen']

    sns.set(color_codes=True)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(X[:, 0], X[:, 1], s=40, alpha=0.9, edgecolors='k', c=y)

    ax.set_xlabel('Concervation_ref')
    ax.set_ylabel('Concervation_alt')
    ax.legend()
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(X_train)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)
    mse = (((prediction - y_test) ** 2).sum()) / len(prediction)

    cf_matrix = confusion_matrix(y_test, prediction)

    print(prediction)

    print(mse)
    print(cf_matrix)
    TN = int(cf_matrix[1][1])
    FN = int(cf_matrix[1][0])
    FP = int(cf_matrix[0][1])
    TP = int(cf_matrix[0][0])

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    # Matthews Correlation Coefficient
    x = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    MCC = (TP * TN - FP * FN) / math.sqrt(x)

    print("De accuray is: ", ACC)
    print("De MCC is: ", MCC)
    print("De FDR is: ", FDR)


def stephantest(train, val, test):
    df = pd.read_csv(train)
    X_train = df[['Concervation_alt', 'Concervation_ref']].values
    y_train = df['Pathogeen']

    df = pd.read_csv(val)
    X_test = df[['Concervation_alt', 'Concervation_ref']].values
    y_test = df['Pathogeen']

    sns.set(color_codes=True)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(X_train[:, 0], X_train[:, 1], s=40, alpha=0.9,
               edgecolors='k', c=y_train)

    ax.set_xlabel('Concervation_ref')
    ax.set_ylabel('Concervation_alt')
    ax.legend()
    plt.show()

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)
    mse = (((prediction - y_test) ** 2).sum()) / len(prediction)

    cf_matrix = confusion_matrix(y_test, prediction)

    print(prediction)

    print(mse)
    print(cf_matrix)
    TN = int(cf_matrix[1][1])
    FN = int(cf_matrix[1][0])
    FP = int(cf_matrix[0][1])
    TP = int(cf_matrix[0][0])

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    # Matthews Correlation Coefficient
    x = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    MCC = (TP * TN - FP * FN) / math.sqrt(x)

    print("De accuracy is: ", ACC)
    print("De MCC is: ", MCC)
    print("De FDR is: ", FDR)


if __name__ == '__main__':
    file = "Files/parsed_data_new.csv"

    train = "train_data_bio_prodict_denormalized_new.csv"
    val = "valid_data_bio_prodict_denormalized_new.csv"
    test = ""

    stephantest(train, val, test)

    # k_nearest_neighbors(file)


def irisoefen():
    iris = datasets.load_iris()

    X = iris.data[:, 2:]
    print(X)
    y = iris.target

    sns.set(color_codes=True)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)

    for index, target_name in enumerate(iris.target_names):
        ax.scatter(X[y == index, 0], X[y == index, 1], s=40, alpha=0.9,
                   edgecolors='k', label=target_name)

    ax.set_xlabel('petal_length')
    ax.set_ylabel('petal_width')
    ax.legend()
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y,
                                                        random_state=42)
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    clf.predict([[4, 1]])

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)

    for index, target_name in enumerate(iris.target_names):
        ax.scatter(X_train[y_train == index, 0], X_train[y_train == index, 1],
                   s=40,
                   alpha=0.9, edgecolors='k', label=target_name)

    ax.plot(4, 1, c='red', alpha=0.9, marker='*', markersize=15)
    ax.set_xlabel('petal_length')
    ax.set_ylabel('petal_width')
    ax.legend()
    plt.show()

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print('Accuracy traindata')
    print(accuracy_score(y_train, y_pred_train))

    print('')
    print('Accuracy testdata')
    print(accuracy_score(y_test, y_pred_test))
