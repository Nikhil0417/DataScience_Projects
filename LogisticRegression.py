import cloudpickle as pickle
data = pickle.load( open( "mnist23.data", "rb" ) )

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
#%matplotlib inline

# plt.figure(figsize=(20,4))
# for index, (image, label) in enumerate(zip(mnist23.data[500:505], mnist23.target[500:505])):
#     plt.subplot(1, 5, index + 1)
#     plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
#     plt.title('Training: %i\n' % label, fontsize = 20)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def sigmoid(z):
	s = 1/(1+np.exp(-z))
	return s

# pca = PCA(n_components=100)
# X_r = pca.fit(data['data']).transform(data['data'])

def accuracy(p,m):
	count = 0
	for i in range(0,p.shape[1]):
		if p[0][i] == 0:
			count = count + 1
	accuracy = (count/m)*100
	return accuracy

X = data['data']
Y = data['target']

l = 0.2

X_train, X_test, y_train, y_test = train_test_split(X,data['target'], random_state = 0)
print(data['data'].shape)
print("")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#Y = y_train
m = 9083
n = 3028
t = 12111
weights = []

class Logit(object):
	def __init__(self):
		pass

	def train(self, X, y):
		# self.X_train = X_train
		# self.y_train = y_train
		self.X = X
		self.Y = Y
	
	def predict(self):
		params = []
		w1 = np.random.random((1,784)) - .5
		b1 = 0
		for dc in range(1,50):
			z = expit(np.dot(w1,X.T) + b1)
			a = sigmoid(z)

			for i in range(0,a.shape[0]):
				for j in range(0,a.shape[1]):
					if (a[i][j]) > 0.5:
						a[i][j] = 3
					else:
						a[i][j] = 2

			dz1 = (a-Y)
			dw1 = np.dot(dz1,X)
			db1 = np.sum(dz1)/t

			w1 = w1 - l*dw1
			b1 = b1 - l*db1
		y_hat = a
		params = [w1,b1, y_hat]
		return params

	def report(self,X_test,y_test):
		out = lr.predict()
		a = out[2]
		count = 0
		err = a-Y
		acc = accuracy(err,m)
		return acc

	def test(self,X_test,y_test):
		out = lr.predict()
		z_test = expit(np.dot(out[0],X_test.T) + out[1])
		a_test = sigmoid(z_test)
		print(a_test.shape[0])
		print(a_test.shape[1])
		for j in range(0,a_test.shape[1]):
			if (a_test[0][j]) > 0.5:
				a_test[0][j] = 3
			else:
				a_test[0][j] = 2

		dz_test = a_test-y_test
		test_accuracy = accuracy(dz_test,n)
		print("Testing accuracy",test_accuracy)
		print("")
		return test_accuracy


lr = Logit()
lr.train(X,Y) #change it to entire dataset
#outs = lr.predict()
#print(outs[1])
#acc = lr.report(X_train,y_train)
#print("Training accuracy",acc)
z = lr.test(X_test,y_test)
print(z)



# def k_validate(X_test,y_test):

# w1 = np.random.random((1,784)) - .5
# print("weights")
# print(w1.shape)
# print("bias")

# 	a1 = a.astype(int)
# 	print("")
# 	J = -(np.sum(Y*(np.log(a1))+((1-Y)*np.log(1-a1))))/m
# 	# J = y*np.log(a)
# 	print(J)
# 	# print(1-a)
	

# 	weights.append((count/m)*100)
# plt.plot(weights)
# plt.show()
# print("final weights",w1.shape)
# print("bias",b1)
# print(len(weights))

# plt.figure(figsize=(20,4))
# for index, (image, label) in enumerate(zip(X_test[9083:9183], y_test[9083:9183])):
# 	#print("image")
#     plt.subplot(1, 5, index + 1)
#     plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
#     plt.title('Training: %i\n' % label, fontsize = 20)
#plt.show()
