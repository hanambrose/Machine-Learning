import math
import numpy as np
from scipy.optimize import minimize
#========================= train data process =================================
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

#--------------------------------Preparing data------------------------------ 

train_data = add_col_1(read_csv('train.csv'))
test_data = add_col_1(read_csv('test.csv'))

train_len = len(train_data)
test_len = len(test_data)
dim_s = len(train_data[0]) -1 # sample dim
#===========================================================================


# compute the kernel/gram matrix
def kernel():
    K_hat_t = np.ndarray([train_len, train_len])
    for i in range(train_len):
        for j in range(train_len):
            K_hat_t[i,j] = (train_data[i][-1])*(train_data[j][-1])*np.inner(train_data[i][0:dim_s],train_data[j][0:dim_s]) 
    return K_hat_t

# dual objective  function
def svm_objective_f(alpha):
    tp1 = alpha.dot(K_hat_) 
    tp2 = tp1.dot(alpha)
    tp3 = -1*sum(alpha)
    return 0.5*tp2 + tp3

def constraint(x):
    return np.inner(x, np.asarray(label_))

# scipy.optimize.minimize to solve optimization problem
def optimized_dual(C):
    bd =(0,C)
    bds = tuple([bd for i in range(train_len)])
    x0 = np.zeros(train_len)
    cons ={'type':'eq', 'fun': constraint}
    sol = minimize(svm_objective_f, x0, method = 'SLSQP', bounds = bds, constraints = cons)
    return [sol.fun, sol.x]

def wt_star(dual_x):
    lenn = len(dual_x)
    ll = []
    for i in range(lenn):
        ll.append(dual_x[i] * train_data[i][-1] * np.asarray(train_data[i][0: dim_s]))
    return sum(ll)
        
def svm_dual(C):
    [sol_f, sol_x] = optimized_dual(C)
    wt = wt_star(sol_x)
    err_1 = predict(wt, train_data)
    err_2 = predict(wt, test_data)
    print('training error=', err_1)
    print('test error=', err_2)
    
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
    
#------------------------- main function ------------------------
K_hat_ = kernel()
label_ = [row[-1] for row in train_data]     
Cs = [100/873, 500/873, 700/873] 
for c in Cs:
    svm_dual(c)
    





