import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,1,200)

y = np.zeros_like(x,dtype = np.int32)

x[0:100] = np.sin(4*np.pi*x)[0:100]

x[100:200] = np.cos(4*np.pi*x)[100:200]

y = 4*np.linspace(0,1,200)+0.5*np.random.randn(200)
label= np.ones_like(x)

label[0:100]=0

plt.scatter(x,y,c=label)

#LDA code
#Covariance for both the classes is same and equal to the total data's covariance
#mean vector for class1
u = np.array([np.mean(x[0:100]), np.mean(y[0:100])])
#mean vector for class2
w = np.array([np.mean(x[100:200]), np.mean(y[100:200])])
#inverse of covariance
lda_cov = np.linalg.inv(np.cov(np.stack((x,y))))
print("Covariances for LDA")
print(lda_cov)
xArr = np.linspace(-4,2,200)
yVal = np.linspace(-1,6,200)
xArr, yVal = np.meshgrid(xArr, yVal)
l_a, l_b, l_c, l_d = lda_cov[0][0], lda_cov[0][1], lda_cov[1][0], lda_cov[1][1]
u1 ,u2 = u[0], u[1]
w1, w2 = w[0], w[1]

plt.contour(xArr,yVal,(xArr*(2*l_a*w1 +l_b*w2 + l_c*w2 - 2*l_a*u1 - l_b*u2 - l_c*u2)
	+ yVal*(2*l_d*w2 + l_c*w1 + l_b*w1 - 2*l_d*u2 - l_c*u1 - l_b*u1) + (u1*u2*(l_b+l_c) + u1*u1*l_a + u2*u2*l_d) - (w1*w2*(l_b+l_c)+w1*w1*l_a + w2*w2*l_d))
	,[1],colors='red',linestyles='dashdot',label='LDA')

#QDA code
qda_cov1 = np.linalg.inv(np.cov(np.stack((x[0:100],y[0:100]))))
qda_cov2 = np.linalg.inv(np.cov(np.stack((x[100:200],y[100:200]))))
print("Covariances of two classes in QDA")
print(qda_cov1)
print(qda_cov2)
det1 = np.linalg.det(qda_cov1)
det2 = np.linalg.det(qda_cov2)
a1, b1, c1, d1 = qda_cov1[0][0], qda_cov1[0][1], qda_cov1[1][0], qda_cov1[1][1]
a2, b2, c2, d2 = qda_cov2[0][0], qda_cov2[0][1], qda_cov2[1][0], qda_cov2[1][1]

plt.contour(xArr,yVal,(xArr*xArr*(a1-a2) + yVal*yVal*(d1-d2) + xArr*yVal*(b1+c1-b2-c2) + xArr*(2*a2*w1 +b2*w2 + c2*w2 - 2*a1*u1 - b1*u2 - c1*u2)
	+ yVal*(2*d2*w2 + c2*w1 + b2*w1 - 2*d1*u2 - c1*u1 - b1*u1) + (u1*u2*(b1+c1) + u1*u1*a1 + u2*u2*d1) - (w1*w2*(b2+c2)+w1*w1*a2 + w2*w2*d2)
	+ (np.log2(det1/det2))/2),[1],colors='blue',linestyles='dashed',label='QDA')
plt.show()
