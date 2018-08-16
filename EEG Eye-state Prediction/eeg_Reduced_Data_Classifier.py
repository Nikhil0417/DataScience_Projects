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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc


# read .csv from provided dataset
csv_filename="EEG_Data.csv"
# df=pd.read_csv(csv_filename,index_col=0)
df=pd.read_csv(csv_filename)
# handle goal attrubte to binary
#popular = df.iloc[:,60] >= 1400	
#unpopular = df.iloc[:,60] < 1400
#df.loc[popular,'class'] = 1
#df.loc[unpopular,'class'] = 0

features=list(df.columns[0:14])
print(type(df[features]))
numpy_news = df[features].as_matrix()
print(type(numpy_news))
#Perform PCA
pca = PCA(n_components=5)
X_PCA = pca.fit(numpy_news).transform(numpy_news)

print(pca.explained_variance_ratio_)
print(type(X_PCA))


acc1 = []
acc2 = []
acc3 = []
acc4 = []
train_size = []
cross_vali = []
cv_rf = []
cv_nb = []
cv_knn = []
for i in range(0,12,2):
# split dataset to 60% training and 40% testing
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_PCA, df.iloc[:,14], test_size=(0.4 - (i*2/100)), random_state=0)
	print(i)
# print X_train.shape, y_train.shape
	print(X_train.shape)
	print(y_train.shape)
	print(X_test.shape)
	print(y_test.shape)
#DecisionTree
	t0=time()
	print("DecisionTree")
	dt = DecisionTreeClassifier(min_samples_split=20,random_state=99)
# dt = DecisionTreeClassifier(min_samples_split=20,max_depth=5,random_state=99)
	clf_dt=dt.fit(X_train,y_train)
	print("Acurracy: ", clf_dt.score(X_test,y_test))
	dect = clf_dt.score(X_test, y_test)
	t1=time()
	print ("time elapsed: ", t1-t0)
	acc1.append(dect)
	train_size.append(y_train.shape[0])
# cross validation for DT
	tt0=time()
	print("cross result========")
	scores = cross_validation.cross_val_score(dt, X_PCA, df.iloc[:,14], cv=10)
	print(scores)
	print(scores.mean())
	tt1=time()
	print("time elapsed: ", tt1-tt0)
	print("\n")
	cross_vali.append(scores.mean())

# #RandomForest
	t2=time()
	print("RandomForest")
	rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
	clf_rf = rf.fit(X_train,y_train)
	print("Acurracy: ", clf_rf.score(X_test,y_test))
	randf = clf_rf.score(X_test,y_test)
	acc2.append(randf)
	t3=time()
	print("time elapsed: ", t3-t2)

# #cross validation for RF
	tt2=time()
	print("cross result========")
	scores = cross_validation.cross_val_score(rf, X_PCA, df.iloc[:,14], cv=10)
	print(scores)
	print(scores.mean())
	tt3=time()
	print("time elapsed: ", tt3-tt2)
	print("\n")
	cv_rf.append(scores.mean())

# #NaiveBayes
	t4=time()
	print("NaiveBayes")
	nb = BernoulliNB()
	clf_nb=nb.fit(X_train,y_train)
	print("Acurracy: ", clf_nb.score(X_test,y_test))
	naive = clf_nb.score(X_test, y_test)
	acc3.append(naive)
	t5=time()
	print("time elapsed: ", t5-t4)

# # cross-validation for NB
	tt4=time()
	print("cross result========")
	scores = cross_validation.cross_val_score(nb, X_PCA, df.iloc[:,14], cv=10)
	print(scores)
	print(scores.mean())
	tt5=time()
	print("time elapsed: ", tt5-tt4)
	print("\n")
	cv_nb.append(scores.mean())

# #KNN accuracy and time elapsed caculation
	t6=time()
	print("KNN")
# # knn = KNeighborsClassifier(n_neighbors=3)
	knn = KNeighborsClassifier()
	clf_knn=knn.fit(X_train, y_train)
	print("Acurracy: ", clf_knn.score(X_test,y_test))
	knear = clf_knn.score(X_test,y_test)
	acc4.append(knear)
	t7=time()
	print("time elapsed: ", t7-t6)

# # cross validation for KNN
	tt6=time()
	print("cross result========")
	scores = cross_validation.cross_val_score(knn, X_PCA, df.iloc[:,14], cv=10)
	print(scores)
	print(scores.mean())
	tt7=time()
	print("time elapsed: ", tt7-tt6)
	print("\n")
	cv_knn.append(scores.mean())

#f, adt, arf, anb, aknn = plt.subplots(4)
# f, ((adt, arf), (anb, aknn)) = plt.subplots(2, 2, sharex='col', sharey='row')
# adt.plot(train_size, acc1)
# adt.plot(train_size, cross_vali)
# adt.set_ylim(ymin=0, ymax=1)

# arf.plot(train_size, acc2)
# arf.plot(train_size, cv_rf)
# arf.set_ylim(ymin=0, ymax=1)

# anb.plot(train_size, acc3)
# anb.plot(train_size, cv_nb)
# anb.set_ylim(ymin=0, ymax=1)

# aknn.plot(train_size, acc4)
# aknn.plot(train_size, cv_knn)
# aknn.set_ylim(ymin=0, ymax=1)

f, axarr = plt.subplots(2, 2)
axarr[0, 0].plot(train_size, acc1)
axarr[0, 0].plot(train_size, cross_vali)
axarr[0, 0].set_ylim(ymin=0, ymax=1)
axarr[0, 0].set_title('DT')

axarr[0, 1].plot(train_size, acc2)
axarr[0, 1].plot(train_size, cv_rf)
axarr[0, 1].set_ylim(ymin=0, ymax=1)
axarr[0, 1].set_title('RF')

axarr[1, 0].plot(train_size, acc3)
axarr[1, 0].plot(train_size, cv_nb)
axarr[1, 0].set_ylim(ymin=0, ymax=1)
axarr[1, 0].set_title('NB')

axarr[1, 1].plot(train_size, acc4)
axarr[1, 1].plot(train_size, cv_knn)
axarr[1, 1].set_ylim(ymin=0, ymax=1)
axarr[1, 1].set_title('K-NN')
# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

plt.show(f)