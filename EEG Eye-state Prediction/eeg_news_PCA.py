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

pca = PCA(n_components=7)
X_PCA = pca.fit(numpy_eyes).transform(numpy_eyes)
#print(type(pca))
#print(len(X_PCA))
print('PCA explained variance ratio : %s' % str(pca.explained_variance_ratio_))
print(pca)
print(type(pca)) #class 'sklearn.decomposition.pca.PCA
print(X_PCA)	#prints 3 principal components
print(type(X_PCA))	#class 'numpy.ndarray'
colors = ['navy', 'green', 'red']
lw = 2
print(np.mean(eyes_target))
print(np.median(eyes_target))
print(np.min(eyes_target))
print(np.max(eyes_target))
plt.scatter(X_PCA[:, 0], X_PCA[:, 1], alpha=.7, lw=lw)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of EEG dataset')

lda = LinearDiscriminantAnalysis(n_components=7)
X_LDA = lda.fit(numpy_eyes,eyes_target).transform(numpy_eyes)

print('LDA explained variance ratio : %s' % str(lda.explained_variance_ratio_))
plt.figure()
plt.scatter(X_LDA[:, 0], X_LDA[:, 1], alpha=.5)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of EEG dataset')
plt.show()

