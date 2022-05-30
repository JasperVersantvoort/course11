#Gijsbert Keja, Roel en Femke mentaal gesteund
#27/05/22
#Dit geavanceerde machine learning algoritme voorspelt of een mutatie
#in een eiwit pathogenic of benign is.


import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import csv

def data_processing(train, test):
    df_train = pandas.read_csv(train)
    df_test = pandas.read_csv(test)
    features = ['Concervation_ref', 'Concervation_alt','Blosumscore']
    x_train = df_train[features].values
    y_train = df_train['Pathogeen']
    names = df_train['Mutation_name']
    x_test = df_test[features].values

    return x_train, y_train, names, x_test


def model_trainen(x_train, y_train):
    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(x_train, y_train)

    return dtree


def model_prediction(dtree):
    lijst_resultaten = []
    for x in range(len(x_test)):
        resultaat = dtree.predict([x_test[x]])
        lijst_resultaten.append(resultaat[0])

    return lijst_resultaten

def csv_generator(x_test, names, lijst_resultaten):
    list_name_pred = []
    for y in range(len(x_test)):
        tijdelijke_lijst = []
        mutation_names = [names[y]]
        pathogeen_benign = [lijst_resultaten[y]]
        tijdelijke_lijst = mutation_names + pathogeen_benign
        list_name_pred.append(tijdelijke_lijst)

    with open("decisiontree_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(list_name_pred)



if __name__ == '__main__':
    train = "train_data_bio_prodict_denormalized_new.csv"
    test = "test_data_bio_prodict_denormalized_new.csv"
    x_train, y_train, names, x_test = data_processing(train, test)
    dtree = model_trainen(x_train, y_train)
    lijst_resultaten = model_prediction(dtree)
    csv_generator(x_test, names, lijst_resultaten)
