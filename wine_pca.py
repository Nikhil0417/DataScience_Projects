"""
Author: Sai Nikhil Reddy Mettupally
Date: 09/25/2017
"""   
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

wine_df = pd.read_csv(
    filepath_or_buffer='http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', 
    header=None, 
    sep=',')                      #fetch the dataset from the repository

wine_data = wine_df.iloc[:,1:15]  #extract the 13 attributes (2nd to 13th columns) into a pandas dataframe
wine_names = wine_df.iloc[:,0]    #extract the class number column into a pandas series
print(wine_data)                  #line 15 and 16 are debug statements to
print(wine_names)                 #check if the dataset is properly sliced
target_arr = [1,2,3]              #target classes of wine

                                  #PCA code segment
pca = PCA(n_components=2)
X_PCA = pca.fit(wine_data).transform(wine_data)
print('PCA explained variance ratio : %s' % str(pca.explained_variance_ratio_))     #gives the 2 eigen values of the principal components
plt.figure()
colors = ['navy', 'green', 'red']
lw = 2
for color, i, target_name in zip(colors, target_arr, target_arr):
    plt.scatter(X_PCA[wine_names == i, 0], X_PCA[wine_names == i, 1], color=color, alpha=.7, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of wine dataset')
plt.show()
