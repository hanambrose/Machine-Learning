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

#--------------------------------Algorithm------------------------------
# Gaussian kernel at each training example
def gauss_kernel_each(s_1, s_2, gamma):
    s1 = np.asarray(s_1)
    s2 = np.asarray(s_2)
    return math.e**(-np.linalg.norm(s1 - s2)**2/gamma)   

# gaussian kernel/gram matrix
def gauss_kernel(gamma):
    K_hat_t = np.ndarray([train_len, train_len])
    for i in range(train_len):
        for j in range(train_len):
            K_hat_t[i,j] = gauss_kernel_each(train_data[i][0:dim_s], train_data[j][0:dim_s], gamma)
    return K_hat_t

# objective function
def svm_objective_f(alpha):
    tp1 = alpha.dot(K_mat_) 
    tp2 = tp1.dot(alpha)
    tp3 = -1*sum(alpha)
    return 0.5*tp2 + tp3

def constraint(x):
    return np.inner(x, np.asarray(label_))

# returns the optimal dual vectors
def optimized_dual(C):
    bd =(0,C)
    bds = tuple([bd for i in range(train_len)])
    x0 = np.zeros(train_len)
    cons ={'type':'eq', 'fun': constraint}
    sol = minimize(svm_objective_f, x0, method = 'SLSQP', bounds = bds, constraints = cons) 
    return [sol.fun, sol.x]

# count the number of support vectors
def count_sv(dual_x):
    ll = []
    for i in range(len(dual_x)):
        if dual_x[i] != 0.0:
            ll.append(i)
    return [np.count_nonzero(dual_x), set(ll)]
                  
def svm_nonlinear(C):
    [sol_f, sol_x] = optimized_dual(C)
    [cnt, gg] = count_sv(sol_x)
    err_1 = predict_ker(sol_x, train_data, gamma)
    err_2 = predict_ker(sol_x, test_data, gamma)
    print('train err=', err_1)
    print('test err=', err_2)
    return [cnt, gg]

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

# optimal dual variables
def predict_ker(dual_x, data, gamma):
    true_label = [row[-1] for row in data]
    pred_seq = [];
    for row in data:
        ll =[]
        for i in range(len(dual_x)):
            ll.append(dual_x[i] * train_data[i][-1] * gauss_kernel_each(train_data[i][0:dim_s], row[0:dim_s], gamma ) )
        pred = sign_func(sum(ll))
        pred_seq.append(pred)
    return error_svm(pred_seq, true_label)

# ================================Report======================================
    
label_ = [row[-1] for row in train_data]  
Cs = [100/873, 500/873, 700/873] 
Gammas =[0.1, 0.5,1,5,100]
for c in Cs:
     for gamma in Gammas:
         print('C=',c, 'gamma=', gamma)
         K_mat_ = gauss_kernel(gamma)
         svm_nonlinear(c)