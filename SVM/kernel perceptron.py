import math
import numpy as np

#========================= data pre-processing =================================
def read_csv(file):
  data = list()
  with open(file, 'r') as f:
    for line in f:
      data.append(line.strip().split(','))
    out = np.array(data).astype(np.float)
    out[:,-1][out[:,-1] == 0] = -1 # change label to 0,1
  return out

def add_col_1 (data):
    col = np.ones(len(data)).reshape(len(data),1)
    out = np.hstack((data[:,:4], col, data[:,4:]))
    return out

def add_count(data):
    col_cnt = np.zeros(len(data)).reshape(len(data),1)
    out = np.hstack((data[:,:6],col_cnt))
    return out
#--------------------------------Preparing data------------------------------ 

train_data = add_count(add_col_1(read_csv('train.csv')))
test_data = add_col_1(read_csv('test.csv'))

train_len = len(train_data)
test_len = len(test_data)
dim_s = len(train_data[0]) -2 # sample dim

#================================kernel perceptron ==========================

def gauss_kernel_each(s_1, s_2, gamma):
    s_1_ = np.asarray(s_1)
    s_2_ = np.asarray(s_2)
    return math.e**(-np.linalg.norm(s_1_ - s_2_)**2/gamma)   

def sign_func(x):
    y = 0
    if x> 0:
        y = 1
    else:
        y=-1
    return y

def single_pred(traindata, sample, gamma):
    temp = sum([row[-1]*row[-2]*gauss_kernel_each(row[0:dim_s], sample, gamma) for row in traindata])
    return sign_func(temp) 

def predict(data, traindata, gamma):
    pred_seq =[]
    for row in data:
        pred_seq.append(single_pred(traindata, row[0:dim_s], gamma))
    return pred_seq

def error_perc(xx,yy):
    cnt = 0
    length =len(xx)
    for i in range(length):
        if xx[i]!= yy[i]:
            cnt = cnt + 1
    return cnt/length        

def kernel_perc(traindata, gamma):
    for row in traindata:
        if row[-2] != single_pred(traindata, row[0:dim_s], gamma):
            row[-1] = row[-1] + 1
    return traindata

# ================================Report======================================
    
Gammas =[0.1, 0.5,1,5,100]
for gamma  in Gammas:
    print('gamma=',gamma)
    train_up = kernel_perc(train_data, gamma)
    pred_seq_train = predict(train_data, train_up, gamma)
    pred_seq_test = predict(test_data, train_up, gamma)
    train_label =[row[-2] for row in train_data]
    test_label = [row[-1] for row in test_data]
    err_train = error_perc(pred_seq_train, train_label)
    err_test = error_perc(pred_seq_test, test_label)
    print('training error =', err_train)
    print('test error =', err_test)





        
        