#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[9]:


#读入数据
data = genfromtxt("delivery.csv",delimiter = ",")


# In[8]:


#切分数据
x_data = data[:,:-1]
y_data = data[:,-1]
print(x_data)
print(y_data)


# In[13]:


#学习率learning rate
lr = 0.0001
#参数
theta0 = 0
theta1 = 0
theta2 = 0
#最大迭代数
epochs = 1000

#最小二乘法
def cost_function(theta0,theta1,theta2,x_data,y_data):
    totalError = 0
    for i in range(0,len(x_data)):
        totalError += (y_data[i]-(theta0 + theta1*x_data[i,0] + theta2*x_data[i,1]))**2
    return totalError/float(len(x_data))/2

#梯度下降法
def gradient_descent_method(x_data,y_data,theta0,theta1,theta2,lr,epochs):
    #计算总数据量
    m = float(len(x_data))
    #循环epochs次
    for i in range(epochs):
        theta0_grad = 0
        theta1_grad = 0
        theta2_grad = 0
        for j in range(0,len(x_data)):
            theta0_grad += (1/m)*((theta0 + theta1*x_data[j,0] + theta2*x_data[j,1]) - y_data[j])
            theta1_grad += (1/m)*((theta0 + theta1*x_data[j,0] + theta2*x_data[j,1]) - y_data[j])*x_data[j,0]
            theta2_grad += (1/m)*((theta0 + theta1*x_data[j,0] + theta2*x_data[j,1]) - y_data[j])*x_data[j,1]
        #更新参数
        theta0 = theta0 - (lr*theta0_grad)
        theta1 = theta1 - (lr*theta1_grad)
        theta2 = theta2 - (lr*theta2_grad)
    return theta0,theta1,theta2


# In[14]:


print("Starting theta0 ={0},theta1 = {1},theta2 = {2}, cost = {3}".
         format(theta0,theta1,theta2,cost_function(theta0,theta1,theta2,x_data,y_data)))
theta0,theta1,theta2 = gradient_descent_method(x_data,y_data,theta0,theta1,theta2,lr,epochs)
print("After{0} iterations theta0 ={1},theta1 = {2},theta2 = {3}, cost = {4}".
         format(epochs,theta0,theta1,theta2,cost_function(theta0,theta1,theta2,x_data,y_data)))


# In[15]:


ax = plt.figure().add_subplot(111,projection = '3d')
ax.scatter(x_data[:,0],x_data[:,1],y_data,c = 'r', marker = 'o', s = 100 )
x0 = x_data[:,0]
x1 = x_data[:,1]
#生成网格矩阵
x0 ,x1 = np.meshgrid(x0,x1)
z = theta0 + theta1*x0 + theta2*x1
# 画3d图
ax.plot_surface(x0,x1,z)
#设置坐标轴
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#显示图像
plt.show()


# In[ ]:




