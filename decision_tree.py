import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

def main():
    os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'
    train_bestand = "train_data_bio_prodict_denormalized_new.csv"
    test_bestand = "test_data_bio_prodict_denormalized_new.csv"
    data = data_maker(train_bestand)
    x_train, x_test, y_train, y_test, feature_cols, X, Y = data_split(data, test_bestand)
    clf = tree_builder(x_train, x_test, y_train, y_test, X, Y)
    tree_visualisation(clf, feature_cols)
    #tree_fit(x_train, y_train, feature_cols)


def data_maker(train_bestand):
    col_names = ["Mutation_name", "Concervation_ref", "Concervation_alt", "Blosumscore", "Pathogeen"]

    # header=0 zorgt ervoor dat de eerste line niet meegenomen wordt,
    # de namen in deze line zijn vervangen met col_names
    #data = pd.read_csv('test.csv', header=0, names=col_names)  # low_memory=False kan bij een error erbij gedaan worden
    data = pd.read_csv(train_bestand, header=0, names=col_names)
    #test_data = pd.read_csv(bestand2, header=0, names=col_names)
    # print(data.head())
    return data

def data_split(data, test_bestand):
    # features waarop de data gesplit wordt (weet nog niet zeker welke ik hiervoor ga gebruiken)
    #feature_cols = ['features', 'conservationGly']
    feature_cols = ["Concervation_ref", "Concervation_alt", "Blosumscore"]
    x = data[feature_cols]
    #y = data.label
    y = data.Pathogeen

    # data split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                        random_state=1)  # 70% training and 30% test
    df2 = pd.read_csv(test_bestand)

    features = ['Concervation_ref', 'Concervation_alt', 'Blosumscore']

    X = df2[features]
    #wss iets anders dan Blosumscore maar idk wat :(
    Y = df2['Blosumscore']

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
