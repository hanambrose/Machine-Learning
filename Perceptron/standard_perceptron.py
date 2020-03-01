import numpy as np

with open('train.csv',mode='r') as f:
    myList_train=[];
    for line in f:
        terms=line.strip().split(',') # 7*N matrix
        myList_train.append(terms)
        
with open('test.csv',mode='r') as f:
    myList_test=[];
    for line in f:
        terms=line.strip().split(',') # 7*N matrix
        myList_test.append(terms)
     
def str_2_flo(mylist):   # convert string to float type and change label to -1, 1
    temp_list = mylist
    for k in range(len(temp_list)):
        temp_list[k][-1] = 2*float(mylist[k][-1])-1
        for i in range(len(temp_list[0])):
            temp_list[k][i] = float(mylist[k][i])            
    return temp_list       

#--------------------------------Preparing data------------------------------   

Num_attr = len(myList_train[0])-1; 
mylist_train = str_2_flo(myList_train)
mylist_test = str_2_flo(myList_test)      
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
    vanilla_perceptron(mylist_train, mylist_test, np.zeros(Num_attr), 0 ,rate,t)
