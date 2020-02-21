# -*- coding: utf-8 -*-
"""
Han Ambrose
Machine Learning - Vanilla Linear Regression
"""
import numpy as np 
import matplotlib.pyplot as plt
import math

###########################################################################

def read_csv(file):
  data = list()
  with open(file, 'r') as f:
    for line in f:
      data.append(line.strip().split(','))   
  return np.array(data).astype(np.float) 
 
#-------------------------------------------------------------------------
  
training_list = read_csv('train.csv')
testing_list = read_csv('test.csv')
#number of attribute
attr = len(training_list[0])-1  

#-------------------------------------------------------------------------

def loss_f(w,dataset):
  loss = 0.5*sum([ (row[-1] - np.inner(w,row[0:7]))**2 for row in dataset ])
  return loss

def grad(w,dataset):
  grad = []
  for j in range(attr):
        grad.append(-sum([ (row[-1] - np.inner(w, row[0:7]))*row[j] for row in dataset]))
  return grad

#-------------------------------------------------------------------------
  
def batch_grad(threshold, rate, w, dataset):
  loss =[]
  while np.linalg.norm(grad(w,dataset)) >= threshold:
    loss.append(loss_f(w,dataset))
    w = w -[rate*x for x in grad(w,dataset)]
  return [w,loss]

def indiv_sgd(threshold, rate, w, dataset, pi):
  flag = 0
  loss_vec =[]
  for x in pi:
     if np.linalg.norm(sgd(w, pi[x], dataset)) <= threshold:
         flag = 1
         return [w, loss_vec, flag]
     loss_vec.append(loss_f(w, dataset))
     w = w - [rate*x for x in sgd(w, pi[x] ,dataset)]     
  return [w, loss_vec, flag]

def sgd(w, sample_idx, dataset):
    s_grad = []
    for j in range(attr):
        s_grad.append(-(dataset[sample_idx][-1]-np.inner(w, dataset[sample_idx][0:7]) )*dataset[sample_idx][j])
    return s_grad

def shuffle(threshold, rate, w, dataset, epoch ):
    loss_all =[]
    for i in range(epoch):
        pi = np.random.permutation(len(training_list))
        [w, loss_vec, flag] = indiv_sgd(threshold, rate, w, dataset, pi)
        if flag == 1:
            return [w, loss_all]
        loss_all = loss_all + loss_vec
    return [w, loss_all]

#######################Running Results####################################
#running Batch Gradient
[wf, lossf] = batch_grad(0.001, 0.01, np.zeros(attr), training_list)
print(wf)
print(loss_f(wf, training_list))
print(loss_f(wf, testing_list))
plt.plot(lossf)
plt.ylabel('loss')
plt.xlabel('Iterations')
plt.title('threshold= 0.001')
plt.show()
#-------------------------------------------------------------------------
#running SGD
[w_sgd, loss_sgd] = shuffle(0.0001, 0.001, np.zeros(attr), training_list, 200)
print(w_sgd)
print(loss_f(w_sgd, training_list))
print(loss_f(w_sgd, testing_list))
plt.plot(loss_sgd)
plt.ylabel('loss')
plt.xlabel('Iterations')
plt.title('threshold= 0.0001, # epochs =200 ')
plt.show()
#-------------------------------------------------------------------------
#Find optimal point
print("###Finding optimal weight vector using analytical form###")
data_list = [row[0:7] for row in training_list]
label_list = [row[-1] for row in training_list]
data_mat = np.array(data_list)
label_mat = np.array(label_list)
X = data_mat.transpose()
 
a = np.linalg.inv(np.matmul(X, X.transpose()))
b = np.matmul(a, X)
c =np.matmul(b, label_mat)
print(c)
print(loss_f(c, training_list))
print(loss_f(c, testing_list))