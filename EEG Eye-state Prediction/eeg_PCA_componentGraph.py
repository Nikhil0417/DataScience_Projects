import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import seaborn as sns; sns.set()

eyes_df = pd.read_csv('EEG_Data.csv')

eyes_target = eyes_df.iloc[:,14] #number of shares (target)
eyes_1000 = eyes_df.iloc[:,0:14] #data
numpy_eyes = eyes_1000.as_matrix() #pandas dataframe to numpy array
#print(type(news_target))
#print(news_target)
print("---------------------------------------------------")
#print(type(news_1000))
#print(news_1000)
#print(type(numpy_news)) #debug statement
sum_comps = []
comps = []
for i in range(1,15):
	x = 0
	pca = PCA(n_components=i)
	X_PCA = pca.fit(numpy_eyes).transform(numpy_eyes)
	x = np.sum(pca.explained_variance_ratio_)
	print(x)
	sum_comps.append(x*100)
	comps.append(i*100/14)
	
plt.bar(comps,sum_comps)
plt.xlabel('Precentage of Features')
plt.ylabel('Percentage of information retained')
plt.title('Feature Retention vs Variance Ratio')
plt.show()