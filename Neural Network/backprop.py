
import numpy as np

"""--------------------------- data pre-processing --------------------------------"""
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
def sign_func(x):
    y = 0
    if x> 0:
        y = 1
    else:
        y=-1
    return y
def error_compute(xx,yy):
    cnt = 0
    length =len(xx)
    for i in range(length):
        if xx[i]!= yy[i]:
            cnt = cnt + 1
    return cnt/length

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0.0, x)
  
def prime(x): # derivative of sigmoid
    return x*(1.0-x)
   
# learning rate
def gamma(t, gamma_0, d): 
    return gamma_0/(1 + (gamma_0/d)*t)

act_func = sigmoid  # choose activation function

class ThreeLayerNet:         
    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2): 
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2 
        self.weight_matrix()
        self.stepsize = gamma
        
    def weight_matrix(self):
        self.W0_mat = np.random.randn(self.hidden_size_1-1, self.input_size)  #switch to np.ones(shape) to initialize to zeros
        self.W1_mat = np.random.randn(self.hidden_size_2-1, self.hidden_size_1)
        self.W2_mat = np.random.randn(self.output_size,self.hidden_size_2)
        
    def forward(self, input_vec):

        input_vec = np.array(input_vec, ndmin=2).T
        h1 = np.dot(self.W0_mat, input_vec)
        h1 = act_func(h1)    
        h2 = np.dot(self.W1_mat, np.concatenate((h1, [[1]]), axis = 0))
        h2 = act_func(h2)   
        output = np.dot(self.W2_mat, np.concatenate((h2, [[1]]), axis = 0))


        return sign_func(output)
        #returns the node z-values at each layer
        
    def train(self, input_vec, true_label, iteration_cnt, gamma_0, d): 

        input_vec =np.array(input_vec, ndmin = 2).T    
        h1 = act_func(np.dot(self.W0_mat, input_vec))  
        h2 = act_func(np.dot(self.W1_mat, np.concatenate((h1, [[1]]), axis = 0)))
        output = np.dot(self.W2_mat, np.concatenate((h2, [[1]]), axis = 0))
        output_error = output - true_label
        grad_w_2 = output_error*(np.concatenate((h2, [[1]]), axis = 0)).T

        h2_error = output_error*(self.W2_mat[0,:][:-1])   
        temp = h2_error*(h2.T) *(1- (h2.T) )  
        tt = np.concatenate((h1, [[1]]), axis = 0)
        grad_w_1 = np.dot( tt, temp).T   

        alpha_vec = self.W2_mat[0,:][:-1]
        beta_vec = prime(h2.T)    
        ab = alpha_vec*beta_vec    
        tpp = np.zeros((self.hidden_size_1 -1,1 ))
        for i in range(self.hidden_size_1 -1):
            tpp[i,0] = output_error*np.inner(ab, self.W1_mat[:,i].T)*prime(h1.T)[0,i]
        grad_w_0 = np.dot(tpp, input_vec.T)

        self.W2_mat = self.W2_mat - gamma(iteration_cnt, gamma_0, d)*grad_w_2
        self.W1_mat = self.W1_mat - gamma(iteration_cnt, gamma_0, d)*grad_w_1 
        self.W0_mat = self.W0_mat - gamma(iteration_cnt, gamma_0, d)*grad_w_0 
        iteration_cnt = iteration_cnt + 1
        return iteration_cnt
                     
        
Ms = [5,10,25,50,100]
gamma_0 = 0.03
d = 2
for M in Ms:
    print("M=", M, "gamma=", gamma_0, "d=", d)
    # create Neural network   
    nn = ThreeLayerNet(input_size = 5, 
            output_size = 1, 
            hidden_size_1 = M, 
            hidden_size_2 = M)
    
    
    # train ThreeLayerNet
    cnt =1
    for i in range(train_len):
        cnt = nn.train(train_data[i][0:dim_s], train_data[i][-1], cnt, gamma_0, d )
    
    # prediction on training data
    y_pred = []
    for i in range(train_len):
        y_true = [row[-1] for row in train_data]
        y_pred.append(nn.forward(train_data[i][0:dim_s]))
    print('train error = ', error_compute(y_pred, y_true))
    
    # prediction on test data
    y_pred_test = []
    for i in range(test_len):
        y_true_test = [row[-1] for row in test_data]
        y_pred_test.append(nn.forward(test_data[i][0:dim_s]))
    print('test error = ', error_compute(y_pred_test, y_true_test))