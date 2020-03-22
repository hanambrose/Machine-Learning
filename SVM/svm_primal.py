import numpy as np


def read_csv(file):
  data = list()
  with open(file, 'r') as f:
    for line in f:
      data.append(line.strip().split(','))
    out = np.array(data).astype(np.float)
    out[:,-1][out[:,-1] == 0] = -1
  return out

def add_col_1 (data):
    col = np.ones(len(data)).reshape(len(data),1)
    out = np.hstack((data[:,:4], col, data[:,4:]))
    return out

#--------------------------------Preparing data------------------------------ 

train_data = add_col_1(read_csv('train.csv'))
test_data = add_col_1(read_csv('test.csv'))

train_len = len(train_data)
test_len = len(test_data)
#===========================================================================
     
def rate_func_1(x, gamma_0, d):
    return gamma_0/(1 + gamma_0*x/d)

def rate_func_2(x, gamma_0):
    return gamma_0/(1 + x)

def sub_grad(curr_wt, sample, iter_cnt, schedule,C, gamma_0,d):

    next_wt = list(np.zeros(len(sample)-1))
    w_0 = curr_wt[0:len(curr_wt)-1]
    w_0.append(0) 
    w_00 = w_0 
    
    if schedule == 1:
        temp_1 = 1- rate_func_1(iter_cnt, gamma_0, d)
        temp_2 = rate_func_1(iter_cnt, gamma_0, d)
        temp_3 = temp_2*C*train_len*sample[-1]
        if sample[-1]*np.inner(sample[0:len(sample)-1], curr_wt) <= 1: 
            next_wt_1 = [x*temp_1 for x in w_00] 
            next_wt_2 = [x*temp_3 for x in sample[0:len(sample)-1]]
            next_wt = [next_wt_1[i] + next_wt_2[i] for i in range(len(next_wt_1))]
        else:
            next_wt = [x*temp_1 for x in w_00]
            
    if schedule == 2:
        temp_1 = 1- rate_func_2(iter_cnt, gamma_0)
        temp_2 = rate_func_2(iter_cnt, gamma_0)
        temp_3 = temp_2*C*train_len*sample[-1]
        if sample[-1]*np.inner(sample[0:len(sample)-1], curr_wt) <= 1: 
            next_wt_1 = [x*temp_1 for x in w_00] 
            next_wt_2 = [x*temp_3 for x in sample[0:len(sample)-1]]
            next_wt = [next_wt_1[i] + next_wt_2[i] for i in range(len(next_wt_1))]
        else:
            next_wt = [x*temp_1 for x in w_00] 
    return next_wt

def loss_func(wt, C, train_data):
    temp=[];
    for i in range(train_len):
        temp.append(max(0, 1- train_data[i][-1]*np.inner(wt, train_data[i][0:len(train_data[0])-1])))
    val = 0.5*np.linalg.norm(wt)**2 + C*sum(temp)
    return val

def svm_single(wt, iter_cnt, permu, train_data,C, schedule, gamma_0, d):
    loss_ = [];
    for i in range(train_len):
        wt = sub_grad(wt, train_data[permu[i]], iter_cnt, schedule,C,gamma_0,d)
        loss_.append(loss_func(wt, C, train_data))
        iter_cnt = iter_cnt + 1
    return [wt, iter_cnt, loss_]


def svm_sgd(wt, T, train_data,C, schedule, gamma_0, d):
    iter_cnt = 1
    loss =[]

    for i in range(T):
        print('T =', i)
        permu = np.random.permutation(train_len)
        [wt, iter_cnt, loss_] = svm_single(wt, iter_cnt, permu, train_data, C, schedule, gamma_0,d)
        loss.extend(loss_)
    return [wt, loss]

def sign_func(x):
    y = 0
    if x> 0:
        y = 1
    else:
        y=-1
    return y

def error_svm(xx,yy):
    cnt = 0
    length =len(xx)
    for i in range(length):
        if xx[i]!= yy[i]:
            cnt = cnt + 1
    return cnt/length

def predict(wt, data):
    pred_seq =[];
    for i in range(len(data)):
        pred_seq.append(sign_func(np.inner(data[i][0:len(data[0])-1], wt)))
    label = [row[-1] for row in data]
    return error_svm(pred_seq, label)        
           

def svm(schedule, T, gamma_0,d):
    Cs = [ x/873 for x in [100,500,700]]
    for c in Cs:
        print('C = ', c)
        wt =list(np.zeros(len(train_data[0])-1)) 
        [ww, loss_val] = svm_sgd(wt, T, train_data, c, schedule, gamma_0, d)
        err_train= predict(ww, train_data)
        err_test = predict(ww, test_data)
        print('Training error:', err_train)
        print('Test error:', err_test)
               
##------------------------------------Report-----------------------------------
T = 100
gamma_0 = 2
d =1

print('Training schedule of learning rate 1')
svm(1, T, gamma_0,d)  

print('Training schedule of learning rate 2')
svm(2, T, gamma_0,d)  
