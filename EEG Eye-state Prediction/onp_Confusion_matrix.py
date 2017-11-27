import itertools
import os
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

from sklearn.metrics import roc_auc_score

#data preprocessing
# read .csv from provided dataset
csv_filename="OnlineNewsPopularity.csv"
df=pd.read_csv(csv_filename)
df2 = df.iloc[::4,:]
popular = df2.iloc[:,60] >= 1400	
unpopular = df2.iloc[:,60] < 1400
df2.loc[popular,'class'] = 1
df2.loc[unpopular,'class'] = 0

features=list(df2.columns[2:60])
numpy_news = df2[features].as_matrix()

acc2 = []
X_train, X_test, y_train, y_test = cross_validation.train_test_split(df2[features], df2.iloc[:,61], test_size=0.32, random_state=0)
#RandomForest
t2=time()
print("Training Data size:",X_train.shape)
print(y_train.shape)
print("Testing Data size:",X_test.shape)
print(y_test.shape)
print("RandomForest")
rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
clf_rf = rf.fit(X_train,y_train)
print("Acurracy: ", clf_rf.score(X_test,y_test))
randf = clf_rf.score(X_test,y_test)
acc2.append(randf)
t3=time()
print("Time elapsed: ", t3-t2)
y_pred = rf.fit(X_train, y_train).predict(X_test)
count = 0
scam = 0
for i in y_pred:
	if(i==1):
		count = count+1
print(count)
for i in y_test:
	if(i==1):
		scam = scam+1
print(scam)
		
def plot_conf_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
                          
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_mat = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_conf_matrix(cnf_mat, classes=['unpopular','popular'], title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_conf_matrix(cnf_mat, classes=['unpopular','popular'], normalize=True, title='Normalized confusion matrix')

plt.show()
