#importing libraries
import os
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from time import time
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt

#data preprocessing
# read .csv from provided dataset
csv_filename="OnlineNewsPopularity.csv"
df=pd.read_csv(csv_filename)
# handle goal attrubte to binary
df2 = df.iloc[::4,:]
popular = df2.iloc[:,60] >= 1400	
unpopular = df2.iloc[:,60] < 1400
df2.loc[popular,'class'] = 1
df2.loc[unpopular,'class'] = 0

features=list(df2.columns[2:60])
numpy_news = df2[features].as_matrix()
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
# for i in range(0,12,2):
	# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_kpca, df2.iloc[:,61], test_size=(0.4 - (i*2/100)), random_state=0)
	# print("Training Data:", 0.6 + i*2/100)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(df2[features], df2.iloc[:,61], test_size=0.32, random_state=0)
#print("Iteration =", i)
print("Training:", X_train.shape)
print("Training:", y_train.shape)
print("Testing:", X_test.shape)
print("Testing:", y_test.shape)
t1_raw = time()
t_raw = t1_raw - t0_raw
#DecisionTree
t0=time()
print("==============DecisionTree Classifier==============")
dt = DecisionTreeClassifier(min_samples_split=20,random_state=99)
clf_dt=dt.fit(X_train,y_train)
print("Acurracy: ", clf_dt.score(X_test,y_test))
dect = clf_dt.score(X_test, y_test)
t1=time()
print ("Time elapsed: ", t1-t0)
acc1.append(dect*100)
train_size.append((y_train.shape[0]/9911)*100)
# cross validation for DT
tt0=time()
print("cross-validation result for DT")
scores_dt = cross_validation.cross_val_score(dt, df2[features], df2.iloc[:,61], cv=5)
print(scores_dt)
print(scores_dt.mean())
tt1=time()
print("Time elapsed: ", tt1-tt0)
print("\n")
cross_vali.append(scores_dt.mean()*100)

# #RandomForest
t2=time()
print("==============RandomForest Classifier==============")
rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
clf_rf = rf.fit(X_train,y_train)
print("Acurracy: ", clf_rf.score(X_test,y_test))
randf = clf_rf.score(X_test,y_test)
acc2.append(randf*100)
t3=time()
print("Time elapsed: ", t3-t2)

# #cross validation for RF
tt2=time()
print("cross-validation result for RF")
scores_rf = cross_validation.cross_val_score(rf, df2[features], df2.iloc[:,61], cv=5)
print(scores_rf)
print(scores_rf.mean())
tt3=time()
print("Time elapsed: ", tt3-tt2)
print("\n")
cv_rf.append(scores_rf.mean()*100)

# #NaiveBayes
t4=time()
print("==============NaiveBayes Classifier==============")
nb = BernoulliNB()
clf_nb=nb.fit(X_train,y_train)
print("Acurracy: ", clf_nb.score(X_test,y_test))
naive = clf_nb.score(X_test, y_test)
acc3.append(naive*100)
t5=time()
print("Time elapsed: ", t5-t4)

# # cross-validation for NB
tt4=time()
print("cross-validation result for NB")
scores_nb = cross_validation.cross_val_score(nb, df2[features], df2.iloc[:,61], cv=5)
print(scores_nb)
print(scores_nb.mean())
tt5=time()
print("Time elapsed: ", tt5-tt4)
print("\n")
cv_nb.append(scores_nb.mean()*100)

# #KNN accuracy and time elapsed caculation
t6=time()
print("==============K-NN Classifier==============")
knn = KNeighborsClassifier()
clf_knn=knn.fit(X_train, y_train)
print("Acurracy: ", clf_knn.score(X_test,y_test))
knear = clf_knn.score(X_test,y_test)
acc4.append(knear*100)
t7=time()
print("Time elapsed: ", t7-t6)

# # cross validation for KNN
tt6=time()
print("cross-validation result for K-NN")
scores_knn = cross_validation.cross_val_score(knn, df2[features], df2.iloc[:,61], cv=5)
print(scores_knn)
print(scores_knn.mean())
tt7=time()
print("Time elapsed: ", tt7-tt6)
print("\n")
cv_knn.append(scores_knn.mean()*100)

#PCA
t0_pca = time()
print("Performing Principal Component Analysis")
pca = PCA(n_components=12)
X_PCA = pca.fit(numpy_news).transform(numpy_news)
print(pca.explained_variance_ratio_)
print(type(X_PCA))

#PCA + Algorithms
acc11 = []
acc21 = []
acc31 = []
acc41 = []
cross_vali1 = []
cv_rf1 = []
cv_nb1 = []
cv_knn1 = []
# split dataset to 60% training and 40% testing
X_train1, X_test1, y_train1, y_test1 = cross_validation.train_test_split(X_PCA, df2.iloc[:,61], test_size=0.32, random_state=0)
t1_pca = time()
t_pca = t1_pca - t0_pca
#DecisionTree
t01=time()
print("==============DecisionTree Classifier==============")
dt1 = DecisionTreeClassifier(min_samples_split=20,random_state=99)
clf_dt1=dt1.fit(X_train1,y_train1)
print("Acurracy: ", clf_dt1.score(X_test1,y_test1))
dect1 = clf_dt1.score(X_test1, y_test1)
t11=time()
print ("Time elapsed: ", t11-t01)
acc11.append(dect1*100)
# cross validation for DT
tt01=time()
print("cross-validation result for DT")
scores_dt1 = cross_validation.cross_val_score(dt1, X_PCA, df2.iloc[:,61], cv=5)
print(scores_dt1)
print(scores_dt1.mean())
tt11=time()
print("Time elapsed: ", tt11-tt01)
print("\n")
cross_vali1.append(scores_dt1.mean()*100)

# #RandomForest
t21=time()
print("==============RandomForest Classifier==============")
rf1 = RandomForestClassifier(n_estimators=100,n_jobs=-1)
clf_rf1 = rf1.fit(X_train1,y_train1)
print("Acurracy: ", clf_rf1.score(X_test1,y_test1))
randf1 = clf_rf1.score(X_test1,y_test1)
acc21.append(randf1*100)
t31=time()
print("Time elapsed: ", t31-t21)

# #cross validation for RF
tt21=time()
print("cross-validation result for RF")
scores_rf1 = cross_validation.cross_val_score(rf1, X_PCA, df2.iloc[:,61], cv=5)
print(scores_rf1)
print(scores_rf1.mean())
tt31=time()
print("Time elapsed: ", tt31-tt21)
print("\n")
cv_rf1.append(scores_rf1.mean()*100)

# #NaiveBayes
t41=time()
print("==============NaiveBayes Classifier==============")
nb1 = BernoulliNB()
clf_nb1=nb1.fit(X_train1,y_train1)
print("Acurracy: ", clf_nb1.score(X_test1,y_test1))
naive1 = clf_nb1.score(X_test1, y_test1)
acc31.append(naive1*100)
t51=time()
print("Time elapsed: ", t51-t41)

# # cross-validation for NB
tt41=time()
print("cross-validation result for NB")
scores_nb1 = cross_validation.cross_val_score(nb1, X_PCA, df2.iloc[:,61], cv=5)
print(scores_nb1)
print(scores_nb1.mean())
tt51=time()
print("Time elapsed: ", tt51-tt41)
print("\n")
cv_nb1.append(scores_nb1.mean()*100)

# #KNN accuracy and time elapsed caculation
t61=time()
print("==============K-NN Classifier==============")
knn1 = KNeighborsClassifier()
clf_knn1=knn1.fit(X_train1, y_train1)
print("Acurracy: ", clf_knn1.score(X_test1,y_test1))
knear1 = clf_knn1.score(X_test1,y_test1)
acc41.append(knear1*100)
t71=time()
print("Time elapsed: ", t71-t61)

# # cross validation for KNN
tt61=time()
print("cross-validation result for K-NN")
scores_knn1 = cross_validation.cross_val_score(knn1, X_PCA, df2.iloc[:,61], cv=5)
print(scores_knn1)
print(scores_knn1.mean())
tt71=time()
print("Time elapsed: ", tt71-tt61)
print("\n")
cv_knn1.append(scores_knn1.mean()*100)

#KPCA
t0_kpca=time()
print("Performing Kernel Principal Component Analysis")
kpca = KernelPCA(n_components=12, kernel="cosine", fit_inverse_transform=True, gamma=10)
X_kpca = kpca.fit_transform(numpy_news)

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
X_train2, X_test2, y_train2, y_test2 = cross_validation.train_test_split(X_kpca, df2.iloc[:,61], test_size=0.32, random_state=0)
t1_kpca = time()
t_kpca = t1_kpca-t0_kpca
#DecisionTree
t02=time()
print("==============DecisionTree Classifier==============")
dt2 = DecisionTreeClassifier(min_samples_split=20,random_state=99)
clf_dt2=dt2.fit(X_train2,y_train2)
print("Acurracy: ", clf_dt2.score(X_test2,y_test2))
dect2 = clf_dt2.score(X_test2, y_test2)
t12=time()
print ("Time elapsed: ", t12-t02)
acc12.append(dect2*100)
# cross validation for DT
tt02=time()
print("cross-validation result for DT")
scores_dt2 = cross_validation.cross_val_score(dt2, X_kpca, df2.iloc[:,61], cv=5)
print(scores_dt2)
print(scores_dt2.mean())
tt12=time()
print("Time elapsed: ", tt12-tt02)
print("\n")
cross_vali.append(scores_dt2.mean()*100)

# #RandomForest
t22=time()
print("==============RandomForest Classifier==============")
rf2 = RandomForestClassifier(n_estimators=100,n_jobs=-1)
clf_rf2 = rf2.fit(X_train2,y_train2)
print("Acurracy: ", clf_rf2.score(X_test2,y_test2))
randf2 = clf_rf2.score(X_test2,y_test2)
acc22.append(randf2*100)
t32=time()
print("Time elapsed: ", t32-t22)

# #cross validation for RF
tt22=time()
print("cross-validation result for RF")
scores_rf2 = cross_validation.cross_val_score(rf2, X_kpca, df2.iloc[:,61], cv=5)
print(scores_rf2)
print(scores_rf2.mean())
tt32=time()
print("Time elapsed: ", tt32-tt22)
print("\n")
cv_rf2.append(scores_rf2.mean()*100)

# #NaiveBayes
t42=time()
print("==============NaiveBayes Classifier==============")
nb2 = BernoulliNB()
clf_nb2=nb2.fit(X_train2,y_train2)
print("Acurracy: ", clf_nb2.score(X_test2,y_test2))
naive2 = clf_nb2.score(X_test2, y_test2)
acc32.append(naive2*100)
t52=time()
print("Time elapsed: ", t52-t42)

# # cross-validation for NB
tt42=time()
print("cross-validation result for NB")
scores_nb2 = cross_validation.cross_val_score(nb2, X_kpca, df2.iloc[:,61], cv=5)
print(scores_nb2)
print(scores_nb2.mean())
tt52=time()
print("Time elapsed: ", tt52-tt42)
print("\n")
cv_nb2.append(scores_nb2.mean()*100)

# #KNN accuracy and time elapsed caculation
t62=time()
print("==============K-NN Classifier==============")
knn2 = KNeighborsClassifier()
clf_knn2=knn2.fit(X_train2, y_train2)
print("Acurracy: ", clf_knn2.score(X_test2,y_test2))
knear2 = clf_knn2.score(X_test2,y_test2)
acc42.append(knear2*100)
t72=time()
print("Time elapsed: ", t72-t62)

# # cross validation for KNN
tt62=time()
print("cross-validation result for K-NN")
scores_knn2 = cross_validation.cross_val_score(knn2, X_kpca, df2.iloc[:,61], cv=5)
print(scores_knn2)
print(scores_knn2.mean())
tt72=time()
print("Time elapsed: ", tt72-tt62)
print("\n")
cv_knn2.append(scores_knn2.mean()*100)

#Bar Graph of Accuracies of different methods
n_groups = 4
raw = [dect*100,randf*100,naive*100,knear*100]
pca_cln = [dect1*100,randf1*100,naive1*100,knear1*100]
kpca_cln = [dect2*100,randf2*100,naive2*100,knear2*100]

fig1, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.20
opacity = 0.8

rects1 = ax.bar(index, raw, bar_width, alpha=opacity, color='#3399FF', label='Raw')
rects2 = ax.bar(index+bar_width, pca_cln, bar_width, alpha=opacity, color='#006600', label='PCA_Cln')				 
rects3 = ax.bar(index+bar_width+bar_width, kpca_cln, bar_width, alpha=opacity, color='#CC0000', label='KPCA_Cln')

ax.set_xlabel('Classification Methods')
ax.set_ylabel('Accuracy in %')
ax.set_title('Accuracy comparison on News Popularity Data')
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
ax.legend()

#Bar graph of time taken to run
fig2, ax = plt.subplots()
raw_time = [t_raw+(t1-t0),t_raw+(t3-t2),t_raw+(t5-t4),t_raw+(t7-t6)]
pca_time = [t_pca+(t11-t01),t_pca+(t31-t21),t_pca+(t51-t41),t_pca+(t71-t61)]
kpca_time = [t_kpca+(t12-t02),t_kpca+(t32-t22),t_kpca+(t52-t42),t_kpca+(t72-t62)]

tower1 = ax.bar(index, raw_time, bar_width, alpha=opacity, color='#3399FF', label='Raw')
tower2 = ax.bar(index+bar_width, pca_time, bar_width, alpha=opacity, color='#006600', label='PCA_Cln')				 
tower3 = ax.bar(index+bar_width+bar_width, kpca_time, bar_width, alpha=opacity, color='#CC0000', label='KPCA_Cln')

ax.set_xlabel('Classification Methods')
ax.set_ylabel('Computation time in seconds')
ax.set_title('Computation time comparison on News Popularity Data')
plt.xticks(index+bar_width, ('DT', 'RF', 'NB', 'K-NN'))
ax.set_ylim(ymin=0, ymax=500)

autolabel(tower1)
autolabel(tower2)
autolabel(tower3)
ax.legend()

#Box plot of 5-fold cross-validation for all the methods
frame01 = pd.DataFrame(scores_dt*100, columns=['DT_raw'])
frame02 = pd.DataFrame(scores_rf*100, columns=['RF_raw'])
frame03 = pd.DataFrame(scores_nb*100, columns=['NB_raw'])
frame04 = pd.DataFrame(scores_knn*100, columns=['K-NN_raw'])

frame11 = pd.DataFrame(scores_dt1*100, columns=['DT_pca'])
frame12 = pd.DataFrame(scores_rf1*100, columns=['RF_pca'])
frame13 = pd.DataFrame(scores_nb1*100, columns=['NB_pca'])
frame14 = pd.DataFrame(scores_knn1*100, columns=['K-NN_pca'])

frame21 = pd.DataFrame(scores_dt2*100, columns=['DT_kpca'])
frame22 = pd.DataFrame(scores_rf2*100, columns=['RF_kpca'])
frame23 = pd.DataFrame(scores_nb2*100, columns=['NB_kpca'])
frame24 = pd.DataFrame(scores_knn2*100, columns=['K-NN_kpca'])
result = pd.concat([frame01,frame11,frame21,frame02,frame12,frame22,frame03,frame13,frame23,frame04,frame14,frame24], axis=1)
color = dict(boxes='#009900', whiskers='#F71313', medians='#003366', caps='#404040')
print(result)
result.plot.box(color=color, sym='r+',rot=45,fontsize='x-small')
plt.xlabel('Classification Methods')
plt.ylabel('Accuracy in %')
plt.title('5-fold Cross Validation on News Popularity Data')
plt.show()
