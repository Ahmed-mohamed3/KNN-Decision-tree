import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv(r"F:\sana 4\Machine learning\Assignment\20198007_20198054_B5_Assignment 1\decision tree\data.csv", header=None)
print(df.head())
print(df.shape)
def handle_missing(df):
    # replace missing values with the most occuring of the column
    df = df.replace('?', np.nan)
    for col in df.columns:
        x = df[col].value_counts().index[0]
        df[col] = df[col].fillna(x)
    return df
# change all n to 0 and y to 1
df = df.replace('n', 0)
df = df.replace('y', 1)
df = handle_missing(df)


y = df[0]
X = df.drop([0], axis=1)
print(X.head())
print(y.head())
print(X.shape)
print(y.shape)

# split data into training and testing
from sklearn.model_selection import train_test_split



tests = [0.5,0.4,0.3,0.2]
accs = [[],[],[],[]]
tree_sizes = [[],[],[],[]]
for _ in range(5):
    seed = np.random.randint(0,100)
    
    for test in range(len(tests)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tests[test], random_state=seed)
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = clf.score(X_test, y_test)
        accs[test].append(acc)
        print(f"Training size: {100- tests[test]*100}% Accuracy: {acc*100}%")
        
        #get the size of the tree
        tree_sizes[test].append(clf.tree_.node_count)
        print (clf.tree_.node_count)
print("Done")

min_accs = []
max_accs = []
avg_accs = []
min_tree_sizes = []
max_tree_sizes = []
avg_tree_sizes = []
for i in range(len(accs)):
    min_accs.append(min(accs[i]))
    max_accs.append(max(accs[i]))
    avg_accs.append(sum(accs[i])/len(accs[i]))
    min_tree_sizes.append(min(tree_sizes[i]))
    max_tree_sizes.append(max(tree_sizes[i]))
    avg_tree_sizes.append(sum(tree_sizes[i])/len(tree_sizes[i]))

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Accuracy and Tree Size')
trainings = [(1-test) * 100 for test in tests]
ax1.plot(trainings, avg_accs, label="Accuracy")
ax1.set(xlabel="Training Size", ylabel="Accuracy")
ax2.plot(trainings, avg_tree_sizes, label="Tree Size")
ax2.set(xlabel="Training Size", ylabel="Tree Size")
plt.show(block = True)
print("Done")