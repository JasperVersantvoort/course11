import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn import tree



def main():
    bestand = 'test.csv.parq'
    bestand2 = "parsed_data_new.csv"
    data = data_maker(bestand)
    x_train, x_test, y_train, y_test, feature_cols, X, Y = data_split(data)
    clf = tree_builder(x_train, x_test, y_train, y_test, X, Y)
    tree_visualisation(clf, feature_cols)
    #tree_fit(x_train, y_train, feature_cols)


def data_maker(bestand):
    #col_names = ["mutation", "conservationPro", "conservationAla", "conservationHis",
                 # "conservationThr", "conservationGln", "conservationTyr", "conservationGly",
                 # "conservationArg", "conservationVal", "consWildType", "conservationGlu",
                 # "conservationMet", "consAervationLys", "conservationIle", "conservationPhe",
                 # "conservationLeu", "conservationAsn", "conservationSer", "conservationAsp",
                 # "conservationCys", "consVariant", "conservationTrp", "source", "label"]

    col_names = ["Mutation_name", "Concervation_ref", "Concervation_alt", "Blosumscore", "Pathogeen"]

    # header=0 zorgt ervoor dat de eerste line niet meegenomen wordt,
    # de namen in deze line zijn vervangen met col_names
    #data = pd.read_csv('test.csv', header=0, names=col_names)  # low_memory=False kan bij een error erbij gedaan worden
    data = pd.read_csv('parsed_data_new.csv', header=0, names=col_names)
    # print(data.head())

    return data


def data_split(data):
    # features waarop de data gesplit wordt (weet nog niet zeker welke ik hiervoor ga gebruiken)
    #feature_cols = ['features', 'conservationGly']
    feature_cols = ["Concervation_ref", "Concervation_alt", "Blosumscore"]
    x = data[feature_cols]
    #y = data.label
    y = data.Pathogeen

    # data split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                        random_state=1)  # 70% training and 30% test
    df2 = pd.read_csv("parsed_data_new.csv")

    features = ['Concervation_ref', 'Concervation_alt', 'Blosumscore']

    X = df2[features]
    Y = df2['Pathogeen']

    return x_train, x_test, y_train, y_test, feature_cols, X, Y


def tree_builder(x_train, x_test, y_train, y_test, X, Y):
    # maak een decision tree classifier
    clf = DecisionTreeClassifier()

    # trainen van de decision tree classifier
    clf = clf.fit(x_train, y_train)
    #
    # voorspelling van de response van de test dataset
    #y_pred = clf.predict(x_test)
    y_pred = clf.predict(X)

    # model accuracy printen:
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Accuracy:", metrics.accuracy_score(Y, y_pred))
    return clf

def tree_visualisation(clf, feature_cols):
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                    feature_names=feature_cols, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('decision_tree.png')
    Image(graph.create_png())

# def tree_fit(x_train, y_train, feature_cols):
#
#     dtree = DecisionTreeClassifier()
#     dtree = dtree.fit(x_train, y_train)
#     data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
#     graph = pydotplus.graph_from_dot_data(data)
#     graph.write_png('mydecisiontree2.png')
#
#     img = pltimg.imread('mydecisiontree2.png')
#     imgplot = plt.imshow(img)
#     plt.show()

main()
