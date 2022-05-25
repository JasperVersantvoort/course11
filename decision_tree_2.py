import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

import pydotplus
import os

# os.environ["PATH"] += os.pathsep + 'C:\\Program Files(x86)\\Graphviz\\bin'

df = pandas.read_csv("Files/parsed_data_new.csv")

features = ['Concervation_ref', 'Concervation_alt', 'Blosumscore']

X = df[features]
y = df['Pathogeen']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')

img=pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show()