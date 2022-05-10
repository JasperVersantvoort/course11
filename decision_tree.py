import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

def decision_tree_maker(bestand):
    col_names = ["mutation", "conservationPro", "conservationAla", "conservationHis",
                 "conservationThr", "conservationGln", "conservationTyr", "conservationGly",
                 "conservationArg", "conservationVal", "consWildType", "conservationGlu",
                 "conservationMet", "conservationLys", "conservationIle", "conservationPhe",
                 "conservationLeu", "conservationAsn", "conservationSer", "conservationAsp",
                 "conservationCys", "consVariant", "conservationTrp", "source", "class"]

    #header=0 zorgt ervoor dat de eerste line niet meegenomen wordt,
    #de namen in deze line zijn vervangen met col_names
    data = pd.read_csv('test.csv', header=0, low_memory=False, names=col_names)
    print(data.head())

def main():
    bestand = 'test.csv.parq'
    decision_tree_maker(bestand)
main()

