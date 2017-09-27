"""
Author: Sai Nikhil Reddy Mettupally
Date: 09/25/2017
"""   
import matplotlib.pyplot as plt
import pandas as pd
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

                                  #LDA code segment
lda = LinearDiscriminantAnalysis(n_components=2)
X_LDA = lda.fit(wine_data,wine_names).transform(wine_data)
print('LDA explained variance ratio : %s' % str(lda.explained_variance_ratio_))
colors = ['navy', 'green', 'red']
plt.figure()
for color, i, target_name in zip(colors, target_arr, target_arr):
    plt.scatter(X_LDA[wine_names == i, 0], X_LDA[wine_names == i, 1], alpha=.7, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of wine dataset')

plt.show()
