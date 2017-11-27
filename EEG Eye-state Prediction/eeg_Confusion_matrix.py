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
from sklearn.decomposition import PCA, KernelPCA

from sklearn.metrics import roc_auc_score


# read .csv from provided dataset
csv_filename="EEG_Data.csv"
df=pd.read_csv(csv_filename)

features=list(df.columns[0:14])
numpy_eyes = df[features].as_matrix()
# pca = PCA(n_components=3)
# X_PCA = pca.fit(numpy_eyes).transform(numpy_eyes)

# print(pca.explained_variance_ratio_)
# print(type(X_PCA))

acc1 = []
acc2 = []
acc3 = []
acc4 = []
train_size = []
cross_vali = []
cv_rf = []
cv_nb = []
cv_knn = []

# kpca = KernelPCA(n_components=10, kernel="cosine", fit_inverse_transform=True, gamma=10)
# #kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
# X_kpca = kpca.fit_transform(numpy_eyes)

#for i in range(0,12,2):
# split dataset to 60% training and 40% testing
X_train, X_test, y_train, y_test = cross_validation.train_test_split(df[features], df.iloc[:,14], test_size=0.32, random_state=0)
#print(i)
# print X_train.shape, y_train.shape
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# #RandomForest
# t2=time()
# print("RandomForest")
# rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
# clf_rf = rf.fit(X_train,y_train)
# print("Acurracy: ", clf_rf.score(X_test,y_test))
# randf = clf_rf.score(X_test,y_test)
# acc2.append(randf)
# t3=time()
# print("time elapsed: ", t3-t2)


t62=time()
print("KNN")
# # knn = KNeighborsClassifier(n_neighbors=3)
knn2 = KNeighborsClassifier()
clf_knn2=knn2.fit(X_train, y_train)
print("Acurracy: ", clf_knn2.score(X_test,y_test))
knear2 = clf_knn2.score(X_test,y_test)
acc4.append(knear2)
t72=time()
print("time elapsed: ", t72-t62)
y_pred = knn2.fit(X_train, y_train).predict(X_test)

# # #NaiveBayes
# t4=time()
# print("NaiveBayes")
# nb = BernoulliNB()
# clf_nb=nb.fit(X_train,y_train)
# print("Acurracy: ", clf_nb.score(X_test,y_test))
# naive = clf_nb.score(X_test, y_test)
# acc3.append(naive)
# t5=time()
# print("time elapsed: ", t5-t4)

# # # cross-validation for NB
# tt4=time()
# print("cross result========")
# scores = cross_validation.cross_val_score(nb, df[features], df.iloc[:,61], cv=10)
# print(scores)
# print(scores.mean())
# tt5=time()
# print("time elapsed: ", tt5-tt4)
# print("\n")
# cv_nb.append(scores.mean())

# # #KNN accuracy and time elapsed caculation
# t6=time()
# print("KNN")
# # # knn = KNeighborsClassifier(n_neighbors=3)
# knn = KNeighborsClassifier()
# clf_knn=knn.fit(X_train, y_train)
# print("Acurracy: ", clf_knn.score(X_test,y_test))
# knear = clf_knn.score(X_test,y_test)
# acc4.append(knear)
# t7=time()
# print("time elapsed: ", t7-t6)

# # # cross validation for KNN
# tt6=time()
# print("cross result========")
# scores = cross_validation.cross_val_score(knn, df[features], df.iloc[:,61], cv=10)
# print(scores)
# print(scores.mean())
# tt7=time()
# print("time elapsed: ", tt7-tt6)
# print("\n")
# cv_knn.append(scores.mean())


def plot_conf_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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
plot_conf_matrix(cnf_matrix, classes=['eyes open','eyes closed'], title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_conf_matrix(cnf_mat, classes=['eyes open','eyes closed'], normalize=True, title='Normalized confusion matrix')

plt.show()
