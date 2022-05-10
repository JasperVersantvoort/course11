import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


def data_maker(bestand):
    col_names = ["mutation", "conservationPro", "conservationAla", "conservationHis",
                 "conservationThr", "conservationGln", "conservationTyr", "conservationGly",
                 "conservationArg", "conservationVal", "consWildType", "conservationGlu",
                 "conservationMet", "conservationLys", "conservationIle", "conservationPhe",
                 "conservationLeu", "conservationAsn", "conservationSer", "conservationAsp",
                 "conservationCys", "consVariant", "conservationTrp", "source", "label"]

    # header=0 zorgt ervoor dat de eerste line niet meegenomen wordt,
    # de namen in deze line zijn vervangen met col_names
    data = pd.read_csv('test.csv', header=0, names=col_names)  # low_memory=False kan bij een error erbij gedaan worden
    #print(data.head())

    return data

def data_split(data):
    # features waarop de data gesplit wordt (weet nog niet zeker welke ik hiervoor ga gebruiken)
    feature_cols = ['mutation', 'label']
    x = data[feature_cols]
    y = data.label

    # data split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                        random_state=1)  # 70% training and 30% test
    return x_train, x_test, y_train, y_test

# def tree_builder(x_train, x_test, y_train, y_test):
#     # maak een decision tree classifier
#     clf = DecisionTreeClassifier()

#     # trainen van de decision tree classifier
#     clf = clf.fit(x_train, y_train)

#     # voorspelling van de response van de test dataset
#     y_pred = clf.predict(x_test)

#     # model accuracy printen:
#     print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


def main():
    bestand = 'test.csv.parq'
    data = data_maker(bestand)
    x_train, x_test, y_train, y_test = data_split(data)
    #werkt nog niet :(
    tree_builder(x_train, x_test, y_train, y_test)
main()

