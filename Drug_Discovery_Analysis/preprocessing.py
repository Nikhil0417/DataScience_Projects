import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


df1 = pd.read_csv("adrb2_properties.csv")

df2 = df1.iloc[0:50,2:55]
#print(df2)

x = df2.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
#This code is to label the samples using the timestamp. May be used after clustering.
#***************************************************************************************************
# names = []
# conformations = []
# df3 = df1.iloc[0:50,0:2]
# print(df3)
# print("-----------------------------------------------------------------")
# for w in df3.iloc[:,0]:
# 	z = w.split("_")

# 	#x = z.split("_")
# 	names.append(z)
# newdf = pd.DataFrame(names)
# #print(newdf[:][1])
# for i in newdf[:][1]:
# 	a = i.split(".")

# 	conformations.append(a)
# cf = pd.DataFrame(conformations)
# #print(cf[0])
# # print(type(z[1]))
# # r = z[1]
# # for i in r:
# # 	x = i.split('.')
# # 	print(x)

# df2.append(cf[0], ignore_index=True)
# print(df2)
#***************************************************************************************************

sum_comps = []
comps = []
for i in range(1,54):
	x = 0
	pca = PCA(n_components=i)
	X_PCA = pca.fit(df).transform(df)
	x = np.sum(pca.explained_variance_ratio_)
	print(i, x, x*100)
	sum_comps.append(x*100)
	comps.append(i*100/58)

print("PCA variance of components:",pca.explained_variance_ratio_)
print(df.shape)
plt.bar(comps,sum_comps)
plt.xlabel('Precentage of Features')
plt.ylabel('Percentage of information retained')
plt.title('Feature Retention vs Variance Ratio')
plt.show()
