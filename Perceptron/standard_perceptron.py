import numpy as np

def read_csv(file):
  data = list()
  with open(file, 'r') as f:
    for line in f:
      data.append(line.strip().split(','))   
  return np.array(data).astype(np.float) 

def change_label(data):
    temp_list = data
    for k in range(len(temp_list)):
        temp_list[k][-1] = 2*float(data[k][-1])-1
    return temp_list    
 
#--------------------------------Preparing data------------------------------ 

training_list = change_label(read_csv('train.csv'))
testing_list = change_label(read_csv('test.csv')) 
Num_attr = len(training_list[0])-1   
     
#--------------------------------perceptron----------------------------------
#perceptron through one epoch
def perceptron(mylist,permu, w, b, rate):  
    for i in range(len(mylist)):
        if (mylist[permu[i]][-1])*(np.inner(w, mylist[permu[i]][0:Num_attr])+b) <=0: #if there is an error then update w
            w = w + [rate*(mylist[permu[i]][-1])*x for x in mylist[permu[i]][0:Num_attr] ]    # arrary summation
            b = b + rate*mylist[permu[i]][-1]*1        
    return [w,b]

#run perceptron through multiple epochs
def epoch_perceptron(train_data, w, b,rate,T):
    for t in range(T):
        permu = np.random.permutation(len(train_data)) # shuffling the data
        #print(w)
        [w, b] = perceptron(train_data, permu, w, b, rate)   
    return [w,b]

#run average prediction error on the test set
def error_perceptron(test_data, w,b):   # using final w from perceptron to compute error for test set
    num_test =len(test_data)
    count = 0
    for i in range(num_test):
        if (test_data[i][-1])*(np.inner(w, test_data[i][0:Num_attr])+b) <=0:
            count +=1;
    return count/num_test

def vanilla_perceptron(train, test, w , b, rate, T):
    [ww,bb] = epoch_perceptron(train, w,b, rate, T)
    err = error_perceptron(test, ww, bb)
    print ('weight is ',ww,'b is ',bb)
    print ('error is ',err)
#--------------------------------------Report--------------------------------
rate = 1  
T = 10
for t in range(1,T+1):
    print('T= ', t)
    vanilla_perceptron(training_list, testing_list, np.zeros(Num_attr), 0 ,rate,t)
