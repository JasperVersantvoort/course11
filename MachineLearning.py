# Thijs Ermens en Gijsbert Keja
# 10-5-2022
# Dit script kan met het parquet file voorspellen of het gen benign of
# pathogeen is

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import csv


def stephantest(train, val):
    """
    Deze functie maakt een grafiek met de 2 datapunten tegenover elkaar. Ook
    evalueert het de model om te kijken wat de MMC, FDR, MSE en accuracy is
    :param train: filename van csv met Mutation_name, Concervation_ref,
    Concervation_alt, Blosumscore, Pathogeen.
    :param val: filename van csv met Mutation_name, Concervation_ref,
    Concervation_alt, Blosumscore, Pathogeen.
    """
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

    steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier(
        n_neighbors=64))]
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Het berekenen van de accuracy als de data wel gescaled is
    a = accuracy_score(y_test, y_pred)
    print("Scaled accuracy is: ", a)

    # Het berekeken van de accuracy als de data niet gescaled is
    knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)
    b = knn_unscaled.score(X_test, y_test)
    print("Unscaled accuracy is: ", b)

    cf_matrix = confusion_matrix(y_test, y_pred)
    print('Confusion matrix: \n', cf_matrix)

    # True Negative
    TN = int(cf_matrix[1][1])
    # False Negative
    FN = int(cf_matrix[1][0])
    # False Positive
    FP = int(cf_matrix[0][1])
    # True Positive
    TP = int(cf_matrix[0][0])

    # Mean Squared Error
    MSE = (((y_pred - y_test) ** 2).sum()) / len(y_pred)
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
    print("De MSE is: ", MSE)


def pipelinescaling(train, test):
    """
    Functie die met de k_nearest_neigbour methode een model maken om met
    de parameters Concervation_alt en Concervation_ref probeert te
    voorspellen of een mutatie pathogeen of benign is. 1 = Pathogeen,
    0 is benign
    :param train: filename van csv met Mutation_name, Concervation_ref,
    Concervation_alt, Blosumscore, Pathogeen.
    :param test: filename van csv met Mutation_name, Concervation_ref,
    Concervation_alt, Blosumscore.
    """
    # Hier wordt de trainingsset ingelezen, dit is de trainingsset die
    # bewerkt is door Jasper en ingelezen door Femke
    df = pd.read_csv(train)
    X_train = df[['Concervation_alt', 'Concervation_ref']].values
    y_train = df['Pathogeen']

    # Hier wordt de validatieset ingelezen. Hier wordt het model op getest
    # hier wordt ook de accuracy op berekend.
    df = pd.read_csv(test)
    X_test = df[['Concervation_alt', 'Concervation_ref']].values
    names = df['Mutation_name']

    # Hier worden de steps van de pipeline gemaakt. In onze pipeline willen
    # we eerst de data schalen met de StandardScaler en vervolgens een model
    # bouwen met de KNeighborsClassifier.
    steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier(
        n_neighbors=64))]
    pipeline = Pipeline(steps)

    # Met GridSearchCV kan de beste accuracy berekend worden. De accuracy
    # gaat wel omhoog als er meer neighbours worden gebruikt. Dit kan
    # kloppen, want hoe meer neighbors hoe meer het voorspeld dat de
    # mutatie benign is

    parameters = {'knn__n_neighbors': np.arange(1, 10)}
    cv = GridSearchCV(pipeline, param_grid=parameters)
    cv.fit(X_train, y_train)
    y_pred = cv.predict(X_test)

    # Hier wordt een bestand weggeschreven met de naam van de mutant en
    # en de score of het pathogeen/benign is
    list_name_pred = []
    for x in range(len(y_pred)):
        name_mutation = [names[x]]
        result = [y_pred[x]]
        tijdelijke_lijst = name_mutation + result
        list_name_pred.append(tijdelijke_lijst)

    with open("knn_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(list_name_pred)


if __name__ == '__main__':
    file = "../parsed_data_new.csv"

    train = "train_data_bio_prodict_denormalized_new.csv"
    val = "valid_data_bio_prodict_denormalized_new.csv"
    test = "test_data_bio_prodict_denormalized_new.csv"

    stephantest(train, val)
    # pipelinescaling(train, test)
