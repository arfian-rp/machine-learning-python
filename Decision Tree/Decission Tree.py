import pandas as pd
import sklearn.model_selection as ms
import sklearn.tree as tree
import sklearn.metrics as met
import pydotplus as pp

df = pd.read_csv('decisiontree_ch6.csv')
encoding = {'mesin': {'bensin':0, "diesel":1},
            'penggerak': {'depan':0, 'belakang':1}}
df.replace(encoding, inplace=True)

X = df.drop(['ID', 'label'], axis=1)
y = df['label']

X_train, X_test, y_train, y_test = ms.train_test_split(X,y, test_size=0.2)

model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('pred: ',y_pred)

score = met.accuracy_score(y_test, y_pred)
print('score: ',score)

# labels = ['mesin','bangku','penggerak']
# dot_data = tree.export_graphviz(model, out_file=None, feature_names=labels, filled=True, rounded=True)
# graph = pp.graph_from_dot_data(dot_data)
# graph.write_png('decisiontree.png')