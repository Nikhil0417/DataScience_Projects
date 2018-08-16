#importing libraries
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
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

#data preprocessing
# read .csv from provided dataset
csv_filename="EEG_Data.csv"
# df=pd.read_csv(csv_filename,index_col=0)
df=pd.read_csv(csv_filename)
# handle goal attrubte to binary
#df2 = df.iloc[::4,:]
#popular = df2.iloc[:,60] >= 1400	
#unpopular = df2.iloc[:,60] < 1400
#df2.loc[popular,'class'] = 1
#df2.loc[unpopular,'class'] = 0

features=list(df.columns[0:14])
numpy_eyes = df[features].as_matrix()
acc1 = []
acc2 = []
acc3 = []
acc4 = []
train_size = []
cross_vali = []
cv_rf = []
cv_nb = []
cv_knn = []

t0_raw = time()
#Original Algorithms
X_train, X_test, y_train, y_test = cross_validation.train_test_split(df[features], df.iloc[:,14], test_size=0.32, random_state=0)
#print("Iteration =", i)
# print X_train.shape, y_train.shape
print("Training:", X_train.shape)
print("Training:", y_train.shape)
print("Testing:", X_test.shape)
print("Testing:", y_test.shape)
t1_raw = time()
t_raw = t1_raw - t0_raw
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
acc1.append(dect*100)
train_size.append((y_train.shape[0]/9911)*100)
# cross validation for DT
tt0=time()
print("cross result========")
scores = cross_validation.cross_val_score(dt, df[features], df.iloc[:,14], cv=5)
print(scores)
print(scores.mean())
tt1=time()
print("time elapsed: ", tt1-tt0)
print("\n")
cross_vali.append(scores.mean()*100)

# #RandomForest
t2=time()
print("RandomForest")
rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
clf_rf = rf.fit(X_train,y_train)
print("Acurracy: ", clf_rf.score(X_test,y_test))
randf = clf_rf.score(X_test,y_test)
acc2.append(randf*100)
t3=time()
print("time elapsed: ", t3-t2)

# #cross validation for RF
tt2=time()
print("cross result========")
scores = cross_validation.cross_val_score(rf, df[features], df.iloc[:,14], cv=5)
print(scores)
print(scores.mean())
tt3=time()
print("time elapsed: ", tt3-tt2)
print("\n")
cv_rf.append(scores.mean()*100)

# #NaiveBayes
t4=time()
print("NaiveBayes")
nb = BernoulliNB()
clf_nb=nb.fit(X_train,y_train)
print("Acurracy: ", clf_nb.score(X_test,y_test))
naive = clf_nb.score(X_test, y_test)
acc3.append(naive*100)
t5=time()
print("time elapsed: ", t5-t4)

# # cross-validation for NB
tt4=time()
print("cross result========")
scores = cross_validation.cross_val_score(nb, df[features], df.iloc[:,14], cv=5)
print(scores)
print(scores.mean())
tt5=time()
print("time elapsed: ", tt5-tt4)
print("\n")
cv_nb.append(scores.mean()*100)

# #KNN accuracy and time elapsed caculation
t6=time()
print("KNN")
# # knn = KNeighborsClassifier(n_neighbors=3)
knn = KNeighborsClassifier()
clf_knn=knn.fit(X_train, y_train)
print("Acurracy: ", clf_knn.score(X_test,y_test))
knear = clf_knn.score(X_test,y_test)
acc4.append(knear*100)
t7=time()
print("time elapsed: ", t7-t6)

# # cross validation for KNN
tt6=time()
print("cross result========")
scores = cross_validation.cross_val_score(knn, df[features], df.iloc[:,14], cv=5)
print(scores)
print(scores.mean())
tt7=time()
print("time elapsed: ", tt7-tt6)
print("\n")
cv_knn.append(scores.mean()*100)

#PCA
t0_pca = time()
pca = PCA(n_components=12)
X_PCA = pca.fit(numpy_eyes).transform(numpy_eyes)

print(pca.explained_variance_ratio_)
print(type(X_PCA))

#PCA + Algorithms
acc11 = []
acc21 = []
acc31 = []
acc41 = []
#train_size1 = []
cross_vali1 = []
cv_rf1 = []
cv_nb1 = []
cv_knn1 = []
# split dataset to 60% training and 40% testing
X_train1, X_test1, y_train1, y_test1 = cross_validation.train_test_split(X_PCA, df.iloc[:,14], test_size=0.32, random_state=0)
t1_pca = time()
t_pca = t1_pca - t0_pca
#DecisionTree
t01=time()
print("DecisionTree")
dt1 = DecisionTreeClassifier(min_samples_split=20,random_state=99)
# dt = DecisionTreeClassifier(min_samples_split=20,max_depth=5,random_state=99)
clf_dt1=dt1.fit(X_train1,y_train1)
print("Acurracy: ", clf_dt1.score(X_test1,y_test1))
dect1 = clf_dt1.score(X_test1, y_test1)
t11=time()
print ("time elapsed: ", t11-t01)
acc11.append(dect1*100)
#train_size1.append((y_train1.shape[0]/9911)*100)
# cross validation for DT
tt01=time()
print("cross result========")
scores = cross_validation.cross_val_score(dt1, X_PCA, df.iloc[:,14], cv=5)
print(scores)
print(scores.mean())
tt11=time()
print("time elapsed: ", tt11-tt01)
print("\n")
cross_vali1.append(scores.mean()*100)

# #RandomForest
t21=time()
print("RandomForest")
rf1 = RandomForestClassifier(n_estimators=100,n_jobs=-1)
clf_rf1 = rf1.fit(X_train1,y_train1)
print("Acurracy: ", clf_rf1.score(X_test1,y_test1))
randf1 = clf_rf1.score(X_test1,y_test1)
acc21.append(randf1*100)
t31=time()
print("time elapsed: ", t31-t21)

# #cross validation for RF
tt21=time()
print("cross result========")
scores = cross_validation.cross_val_score(rf1, X_PCA, df.iloc[:,14], cv=5)
print(scores)
print(scores.mean())
tt31=time()
print("time elapsed: ", tt31-tt21)
print("\n")
cv_rf1.append(scores.mean()*100)

# #NaiveBayes
t41=time()
print("NaiveBayes")
nb1 = BernoulliNB()
clf_nb1=nb1.fit(X_train1,y_train1)
print("Acurracy: ", clf_nb1.score(X_test1,y_test1))
naive1 = clf_nb1.score(X_test1, y_test1)
acc31.append(naive1*100)
t51=time()
print("time elapsed: ", t51-t41)

# # cross-validation for NB
tt41=time()
print("cross result========")
scores = cross_validation.cross_val_score(nb1, X_PCA, df.iloc[:,14], cv=5)
print(scores)
print(scores.mean())
tt51=time()
print("time elapsed: ", tt51-tt41)
print("\n")
cv_nb1.append(scores.mean()*100)

# #KNN accuracy and time elapsed caculation
t61=time()
print("KNN")
# # knn = KNeighborsClassifier(n_neighbors=3)
knn1 = KNeighborsClassifier()
clf_knn1=knn1.fit(X_train1, y_train1)
print("Acurracy: ", clf_knn1.score(X_test1,y_test1))
knear1 = clf_knn1.score(X_test1,y_test1)
acc41.append(knear1*100)
t71=time()
print("time elapsed: ", t71-t61)

# # cross validation for KNN
tt61=time()
print("cross result========")
scores = cross_validation.cross_val_score(knn1, X_PCA, df.iloc[:,14], cv=5)
print(scores)
print(scores.mean())
tt71=time()
print("time elapsed: ", tt71-tt61)
print("\n")
cv_knn1.append(scores.mean()*100)

#KPCA
t0_kpca=time()
kpca = KernelPCA(n_components=12, kernel="rbf", fit_inverse_transform=True, gamma=10)
#kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
X_kpca = kpca.fit_transform(numpy_eyes)

#KPCA + Algorithms
acc12 = []
acc22 = []
acc32 = []
acc42 = []
#train_size2 = []
cross_vali2 = []
cv_rf2 = []
cv_nb2 = []
cv_knn2 = []
# split dataset to 60% training and 40% testing
X_train2, X_test2, y_train2, y_test2 = cross_validation.train_test_split(X_kpca, df.iloc[:,14], test_size=0.32, random_state=0)
t1_kpca = time()
t_kpca = t1_kpca-t0_kpca
#DecisionTree
t02=time()
print("DecisionTree")
dt2 = DecisionTreeClassifier(min_samples_split=20,random_state=99)
# dt = DecisionTreeClassifier(min_samples_split=20,max_depth=5,random_state=99)
clf_dt2=dt2.fit(X_train2,y_train2)
print("Acurracy: ", clf_dt2.score(X_test2,y_test2))
dect2 = clf_dt2.score(X_test2, y_test2)
t12=time()
print ("time elapsed: ", t12-t02)
acc12.append(dect2*100)
#train_size2.append((y_train2.shape[0]/9911*100))
# cross validation for DT
tt02=time()
print("cross result========")
scores = cross_validation.cross_val_score(dt2, X_kpca, df.iloc[:,14], cv=5)
print(scores)
print(scores.mean())
tt12=time()
print("time elapsed: ", tt12-tt02)
print("\n")
cross_vali.append(scores.mean()*100)

# #RandomForest
t22=time()
print("RandomForest")
rf2 = RandomForestClassifier(n_estimators=100,n_jobs=-1)
clf_rf2 = rf2.fit(X_train2,y_train2)
print("Acurracy: ", clf_rf2.score(X_test2,y_test2))
randf2 = clf_rf2.score(X_test2,y_test2)
acc22.append(randf2*100)
t32=time()
print("time elapsed: ", t32-t22)

# #cross validation for RF
tt22=time()
print("cross result========")
scores = cross_validation.cross_val_score(rf2, X_kpca, df.iloc[:,14], cv=5)
print(scores)
print(scores.mean())
tt32=time()
print("time elapsed: ", tt32-tt22)
print("\n")
cv_rf2.append(scores.mean()*100)

# #NaiveBayes
t42=time()
print("NaiveBayes")
nb2 = BernoulliNB()
clf_nb2=nb2.fit(X_train2,y_train2)
print("Acurracy: ", clf_nb2.score(X_test2,y_test2))
naive2 = clf_nb2.score(X_test2, y_test2)
acc32.append(naive2*100)
t52=time()
print("time elapsed: ", t52-t42)

# # cross-validation for NB
tt42=time()
print("cross result========")
scores = cross_validation.cross_val_score(nb2, X_kpca, df.iloc[:,14], cv=5)
print(scores)
print(scores.mean())
tt52=time()
print("time elapsed: ", tt52-tt42)
print("\n")
cv_nb2.append(scores.mean()*100)

# #KNN accuracy and time elapsed caculation
t62=time()
print("KNN")
# # knn = KNeighborsClassifier(n_neighbors=3)
knn2 = KNeighborsClassifier()
clf_knn2=knn2.fit(X_train2, y_train2)
print("Acurracy: ", clf_knn2.score(X_test2,y_test2))
knear2 = clf_knn2.score(X_test2,y_test2)
acc42.append(knear2*100)
t72=time()
print("time elapsed: ", t72-t62)

# # cross validation for KNN
tt62=time()
print("cross result========")
scores = cross_validation.cross_val_score(knn2, X_kpca, df.iloc[:,14], cv=5)
print(scores)
print(scores.mean())
tt72=time()
print("time elapsed: ", tt72-tt62)
print("\n")
cv_knn2.append(scores.mean()*100)

#Bar Graph of Accuracies of different methods
n_groups = 4
raw = [dect*100,randf*100,naive*100,knear*100]
pca_cln = [dect1*100,randf1*100,naive1*100,knear1*100]
kpca_cln = [dect2*100,randf2*100,naive2*100,knear2*100]

# daata = {'Decision Tree': dect*100,'Random Forest': randf*100,'Naive Bayes': naive*100,'K-NN': knear*100,
		# 'Decision Tree': dect1*100,'Random Forest': randf1*100,'Naive Bayes': naive1*100,'K-NN': knear1*100,
		# 'Decision Tree': dect2*100,'Random Forest': randf2*100,'Naive Bayes': naive2*100,'K-NN': knear2*100}

# frame = pd.DataFrame(daata)
# frame2 = pd.DataFrame(daata, columns=['Original','PCA+Classification','KPCA+Classification'])
# frame.plot.bar()
#n_groups = 3
#desc = [dect*100,dect1*100,dect2*100]
#forest = [randf*100,randf1*100,randf2*100]
#bayes = [naive*100,naive1*100,naive2*100]
#neighbor = [knear*100,knear1*100,knear2*100]

fig1, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.20
opacity = 0.8

rects1 = ax.bar(index, raw, bar_width, alpha=opacity, color='#3399FF', label='Raw')
rects2 = ax.bar(index+bar_width, pca_cln, bar_width, alpha=opacity, color='#006600', label='PCA')				 
rects3 = ax.bar(index+bar_width+bar_width, kpca_cln, bar_width, alpha=opacity, color='#CC0000', label='KPCA')

#rects1 = plt.bar(index, desc, bar_width, alpha=opacity, color='#3399FF', label='DT')
#rects2 = plt.bar(index+bar_width, forest, bar_width, alpha=opacity, color='#006600', label='RF')				
#rects3 = plt.bar(index+(bar_width*2), bayes, bar_width, alpha=opacity, color='#FFFF00', label='NB')
#rects4 = plt.bar(index+(bar_width*3), neighbor, bar_width, alpha=opacity, color='#CC0000', label='K-NN')

ax.set_xlabel('Classification Methods')
ax.set_ylabel('Accuracy in %')
ax.set_title('Accuracy comparison on EEG Eye Data')
plt.xticks(index+bar_width, ('DT', 'RF', 'NB', 'K-NN'))
ax.set_ylim(ymin=0, ymax=100)

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height,
                '%.2f' % rect.get_height(),
                ha='center', va='bottom', weight='bold', size='xx-small')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
# #plt.xticks(index+(bar_width*4), ('Classification', 'PCA + Classification', 'KPCA + Classification'))
ax.legend()
plt.figure()

#Bar graph of time taken to run
fig2, ax = plt.subplots()
raw_time = [t_raw+(t1-t0),t_raw+(t3-t2),t_raw+(t5-t4),t_raw+(t7-t6)]
pca_time = [t_pca+(t11-t01),t_pca+(t31-t21),t_pca+(t51-t41),t_pca+(t71-t61)]
kpca_time = [t_kpca+(t12-t02),t_kpca+(t32-t22),t_kpca+(t52-t42),t_kpca+(t72-t62)]

tower1 = ax.bar(index, raw_time, bar_width, alpha=opacity, color='#3399FF', label='Raw')
tower2 = ax.bar(index+bar_width, pca_time, bar_width, alpha=opacity, color='#006600', label='PCA')				 
tower3 = ax.bar(index+bar_width+bar_width, kpca_time, bar_width, alpha=opacity, color='#CC0000', label='KPCA')

ax.set_xlabel('Classification Methods')
ax.set_ylabel('Computation time in seconds')
ax.set_title('Computation time comparison on EEG Eye Data')
plt.xticks(index+bar_width, ('DT', 'RF', 'NB', 'K-NN'))

autolabel(tower1)
autolabel(tower2)
autolabel(tower3)
ax.legend()
plt.show()