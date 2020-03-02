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
#---------------------------------------------------------------------------
def CCount_update(ll): # only keep weights that has count and remove ones with no count
    temp =[];
    for i in range(len(ll)):
        if ll[i]!= 0:
            temp.append(ll[i])
    return temp
            
# wm--(w,b) matrix, CCount--the number of predictions made by wm   
def voted_perceptron(train_data, w, b, rate, T):
    data_len = len(train_data)
    permu = [];   
    for i in range(T):
        permu += np.random.permutation(data_len).tolist()
    CCount =np.zeros(T*data_len).tolist()
    wm= []
    m =0
    for i in range(T*data_len):
        if (train_data[permu[i]][-1])*( np.inner(w, train_data[permu[i]][0:Num_attr])+b ) <= 0:
            w = w + [rate *(train_data[permu[i]][-1])*x for x in train_data[permu[i]][0:Num_attr]] 
            b= b + rate*train_data[permu[i]][-1]   
            m +=1
            row = np.append(w,b)
            wm.append(row)
            CCount[m]=1                
        if (train_data[permu[i]][-1])*( np.inner(w, train_data[permu[i]][0:Num_attr])+b ) > 0:
            CCount[m] += 1       
    return [wm, CCount_update(CCount)]

def sign_func(x):
    sign=0
    if x >0:
        sign=1
    else:
        sign=-1
    return sign

def each_output(wm,c):
    temp =[]
    for i in range(len(wm)):
        tt = wm[i].tolist() + [c[i]]
        temp.append(tt)
    return temp

#using the set of weight vector and c to predict each test example
def voted_perc_error(test_data, wm,c):
    wm_c =each_output(wm,c)
    pred_seq =[]
    for i in range(len(test_data)):
       pred_seq.append(sign_func(sum( [(row[-1])*sign_func(np.inner(test_data[i][0:Num_attr], row[0:Num_attr] )+row[Num_attr]) for row in wm_c])))
    count =0
    for i in range(len(test_data)):
        if pred_seq[i] != test_data[i][-1]:
            count+=1
    return count/len(test_data)

#----------------------------------Report------------------------------------
rate = 1    
T =10

for t in range(1,T+1):
    print('T= ', t)
    [wm,c]=voted_perceptron(training_list,np.zeros(Num_attr), 0, rate, t)
    print('count list')
    print(c)
    print('weight and counts')
    print(np.round(each_output(wm,c),2))
    err = voted_perc_error(testing_list, wm,c)
    print('error',err)