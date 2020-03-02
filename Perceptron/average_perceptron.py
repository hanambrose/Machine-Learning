import numpy as np
np.set_printoptions(threshold=5, suppress = True)

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

#--------------------------------average Perceptron--------------------------
                
# avg is the avg weight
def avg_perceptron(train_data, w, b, rate, T):
    data_len = len(train_data)
    permu = [];   # length = T*len(train_data))
    for i in range(T):
        permu += np.random.permutation(data_len).tolist()
    avg = np.zeros(Num_attr)
    bias= 0
    for i in range(T*data_len):
        if (train_data[permu[i]][-1])*( np.inner(w, train_data[permu[i]][0:Num_attr])+b ) <= 0:
            w = w + [rate *(train_data[permu[i]][-1])*x for x in train_data[permu[i]][0:Num_attr]] 
            b= b + rate*train_data[permu[i]][-1]  
            avg = avg+ w
            bias= bias+b
    return [avg,bias]

def sign_func(x):
    sign=0
    if x >0:
        sign=1
    else:
        sign=-1
    return sign

def avg_perc_error(test_data, avg,bias):
    pred_seq =[]
    for i in range(len(test_data)):
       pred_seq.append(sign_func(np.inner(avg, test_data[i][0:Num_attr])+ bias))
    count =0
    for i in range(len(test_data)):
        if pred_seq[i] != test_data[i][-1]:
            count+=1
    return count/len(test_data)
 
#--------------------------------Report--------------------------------------

rate = 1    
T =10
for t in range(1,T+1):
    print('T= ', t)
    [avg_w, bias_term] = avg_perceptron(training_list, np.zeros(Num_attr), 0, rate, t)
    print('avg_weight:',avg_w,'b:', bias_term)
    err = avg_perc_error(testing_list, avg_w, bias_term)
    print('error:',err)