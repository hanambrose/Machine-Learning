# Adaboost algorithm
# only for stumps, the 'used attrs can not be used again ' issue is not fixed here
import math
import statistics
import numpy as np
import matplotlib.pyplot as plot
#-----------------train data process------------------------------------
with open('train.csv',mode='r') as f:
    training_list=[];
    for line in f:
        data_point=line.strip().split(',') # 7*N matrix
        training_list.append(data_point)

num_set={0,5,9,11,12,13,14} # indices of numeric attr   
def str_2_flo(mylist):
    temp_list = mylist
    for k in range(len(temp_list)):
        for i in {0,5,9,11,12,13,14}:
            temp_list[k][i] = float(mylist[k][i])
    return temp_list

training_list = str_2_flo(training_list)

obj={0:0,5:5,9:9,11:11,12:12,13:13,14:14}
for i in obj:
    obj[i] = statistics.median([row[i] for row in training_list])
    
for row in training_list:
    for i in obj:
        if row[i] >= obj[i]:
            row[i] = 'yes'
        else:
            row[i] = 'no'                     
#--------------------test data process--------------------------------
with open('test.csv',mode='r') as f:
    testing_list=[];
    for line in f:
        data_point=line.strip().split(',') # 7*N matrix
        testing_list.append(data_point)

testing_list = str_2_flo(testing_list)
for i in obj:
    obj[i] = statistics.median([row[i] for row in testing_list])
#binary quantization of numerical attributes
for row in testing_list:
    for i in obj:
        if row[i] >= obj[i]:
            row[i] = 'yes'
        else:
            row[i] = 'no'  
#------------------------------------------------------------------------------
Attr_dict ={'age':['yes','no'],
             'job':['admin.','unknown','unemployed','management',
                    'housemaid','entrepreneur','student','blue-collar',
                    'self-employed','retired','technician','services'],
                    'martial':['married','divorced','single'],
                    'education':['unknown','secondary','primary','tertiary'],
                     'default':['yes','no'],
                     'balance':['yes','no'],
                     'housing':['yes','no'],
                     'loan':['yes','no'],
                     'contact':['unknown','telephone','cellular'],
                     'day':['yes','no'],
                     'month':['jan', 'feb', 'mar', 'apr','may','jun','jul','aug','sep','oct', 'nov', 'dec'],
                     'duration': ['yes','no'],
                     'campaign':['yes','no'],
                     'pdays':['yes','no'],
                     'previous':['yes','no'],
                     'poutcome':[ 'unknown','other','failure','success']}

Attr_set = set(key for key in Attr_dict)   # the set of all attr_indexsible attr.s

def attr_index(attr):
    attr_dict_int = {'age': 0, 
                     'job':1,
                     'martial':2,
                     'education':3,
                     'default':4,
                     'balance':5,
                     'housing':6,
                     'loan':7,
                     'contact':8,
                     'day':9,
                     'month':10,
                     'duration':11,
                     'campaign':12,
                     'pdays':13,
                     'previous':14,
                     'poutcome':15,
                     'y':16}    
    attr_index = attr_dict_int[attr]
    return attr_index      
 
#----------------------------------------------------------------------------  
def branches_list(attr):
    obj={}
    for attr_val in Attr_dict[attr]:
        obj[attr_val]=[]
    return obj   # dict type with list value type

def attr_list(attr):
    obj={}
    for attr_val in attr:
        obj[attr_val]=0
    return obj    # dict type with float value type   dict=(key,value) 
#----------------------------------------------------------------------------   
def info_gain(branches, labels):
    #N_ins= float(sum([len(branches[attr_val]) for attr_val in branches])) 
    Q = 0.0   #total weights
    tp =0.0
    for attr_val in branches:
        tp = sum([row[-1] for row in branches[attr_val]])
        Q = Q + tp        
    exp_ent = 0.0
    for attr_val in branches:
        size = float(len(branches[attr_val]))
        if size == 0:
            continue    # jump this iteration
        score = 0
        q = sum([row[-1] for row in branches[attr_val]])
        for class_val in labels:
#           p = [row[-3] for row in branches[attr_val]].count(class_val) / size   ###            
            p = sum([row[-1] for row in branches[attr_val] if row[-2] == class_val])/q  #sum up the weights
            if p==0:
                temp=0
            else:
                temp=p*math.log2(1/p)
            score +=temp 
#        exp_ent += score* (size / N_ins)
        exp_ent += score* sum([row[-1] for row in branches[attr_val]])/Q #total weights of a subset
    return exp_ent          
#--------------------------------------------------------------------------
#dataset: list type
# return an dict with subsets of samples corres. to vlaues of attr.
def data_split(attr, dataset):
    branch_obj=branches_list(attr)  # this may result in empty dict elements 
    for row in dataset:
        for attr_val in Attr_dict[attr]:
           if row[attr_index(attr)] == attr_val:
               branch_obj[attr_val].append(row)
    return branch_obj
#----------------------------------------------------------------------------

def best_spl(dataset):
    if dataset==[]:
        return 
    label_values = list(set(row[-2] for row in dataset)) 
    metric_obj = attr_list(Attr_dict)
    for attr in Attr_dict:
        branches = data_split(attr, dataset)
        metric_obj[attr] = info_gain(branches, label_values)             # change metric here
    best_attr = min(metric_obj, key=metric_obj.get)
    best_branches = data_split(best_attr, dataset)  
    return {'best_attr':best_attr, 'best_branches':best_branches}
#----------------------------------------------------------------------------    
# returns the majority label within 'group' (list type)
def leaf_node_label(group):
    majority_labels = [row[-2] for row in group]    # we deal with data appended with weight 
    return max(set(majority_labels), key=majority_labels.count)
#----------------------------------------------------------------------------
def if_node_divisible(branch_obj):
    non_empty_indices=[key for key in branch_obj if not (not branch_obj[key])]
    if len(non_empty_indices)==1:
        return False
    else:
        return True
#-------------------------------------------------------------------------
def internal_node(node, max_depth, curr_depth):
#    if not if_node_divisible(node['best_branches']):  #only one non-empty branch
#        # what if all elements in node 
#        for key in node['best_branches']:
#            if  node['best_branches'][key]!= []: #and ( not isinstance(node['best_branches'][key],str)): 
#                 node[key] = leaf_node_label(node['best_branches'][key])
#            else:
#                node[key] = leaf_node_label(sum(node['best_branches'].values(),[])) 
#        return
    if curr_depth >= max_depth:
        for key in node['best_branches']:
            if  node['best_branches'][key]!= []: #and ( not isinstance(node['best_branches'][key],str)):
                # extract nonempty branches
                node[key] = leaf_node_label(node['best_branches'][key])   
            else:
                node[key] = leaf_node_label(sum(node['best_branches'].values(),[]))
        return  
    for key in node['best_branches']:
        if node['best_branches'][key]!= []: #and ( not isinstance(node['best_branches'][key],str)):
            node[key] = best_spl(node['best_branches'][key]) 
            internal_node(node[key], max_depth, curr_depth + 1)
        else:
            node[key] = leaf_node_label(sum(node['best_branches'].values(),[]))  

    
#----------------------------------------------------------------------------
def tree_build(train_data, max_depth):
	root = best_spl(train_data)               
	internal_node(root, max_depth, 1)
	return root
#----------------------------------------------------------------------------
#test if an instance belongs to a node recursively
def label_predict(node, inst):
    if isinstance(node[inst[attr_index(node['best_attr'])]],dict):
        return label_predict(node[inst[attr_index(node['best_attr'])]],inst)
    else:
        return node[inst[attr_index(node['best_attr'])]]   #leaf node

#sign function
def sign_func(val):
    if val > 0:
        return 1.0
    else:
        return -1.0 
#------return the true label and predicted result using stump 'tree'-----
def label_return(dataset,tree):
    true_label = []
    pred_seq = []   # predicted sequence
    for row in dataset:
        true_label.append(row[-2])    
        pre = label_predict(tree, row)
        pred_seq.append(pre)
    return [true_label, pred_seq]
    
# create dict with n keys and list element
def list_obj(n):
    obj={}
    for i in range(n):
        obj[i] = []
    return obj
#-------------------------convert label to binaries---------------------
def bin_quan(llist): # convert yes and no to 1 and -1
    bin_list =[]
    for i in range(len(llist)):
        if llist[i] == 'yes':
            bin_list.append(1.0)
        else:
            bin_list.append(-1.0)
    return bin_list
def wt_update(curr_wt, vote, bin_true, bin_pred): #update wt  
    next_wt=[]  
    for i in range(len(bin_true)):
        next_wt.append(curr_wt[i]*math.e**(- vote*bin_true[i]*bin_pred[i]))
    next_weight = [x/sum(next_wt) for x in next_wt]
    return next_weight

#----------------------------------------------------------------------
def wt_append(mylist, weights):
    for i in range(len(mylist)):
        mylist[i].append(weights[i]) 
    return mylist 
# replace updated weight to the entire dataset
def wt_update_2_data(data, weight):
    for i in range(len(data)):
        data[i][-1] = weight[i]
    return data
#-----------------------------------------------------------
# indiv_pred: individual stump prediction result
def ada_fin_pred(indiv_pred, vote, data_len, iteration):
    fin_pred = []
    for j in range(data_len):
        score = sum([indiv_pred[i][0][j]*vote[i] for i in range(iteration)])
        fin_pred.append(sign_func(score))
    return fin_pred
#-----------------------------weighted error------------------     
def error_at_t(true_label, predicted, weights):
    count = 0  # correct predication count
    for i in range(len(true_label)):
        if true_label[i] != predicted[i]:
            count += weights[i]
    return count
#    return count / float(len(true_label)) * 100.0
def total_error(_true_lb, _pred_lb): 
    count = 0
    size = len(_true_lb)
    for i in range(size):
        if _true_lb[i] != _pred_lb[i]:
            count += 1
    return count/size 
#===================================boosting starts here================================           
# _delta : small item added to err to avoid infinite vote
def ada_boost(T_iter, _delta, train_data, stump):
    pred_result = list_obj(T_iter)  
    votes = []
    weights = [row[-1] for row in train_data]
    for i in range(T_iter):
        tree = tree_build(train_data, stump)    # train stumps
        [pp_true, qq_pred] = label_return(train_data, tree)   # prediction result before adaboost
        pred_result[i].append(bin_quan(qq_pred))
        err = error_at_t(pp_true, qq_pred, weights)  #error at t round
        votes.append( 0.5*math.log((1-err)/err ))   #final vote of each stump
        weights = wt_update(weights, 0.5*math.log((1-err)/err ), bin_quan(pp_true), bin_quan(qq_pred))
        train_data = wt_update_2_data(train_data, weights) 
    return [pred_result, votes, weights]

# ==================================Report=====================================
   
W_1 = np.ones(len(training_list))/len(training_list)   # wt initialization
training_list = wt_append(training_list, W_1) # append initial weight to training data

true_label_bin = bin_quan([row[-2] for row in training_list]) #get true label from training
true_label_bin_test = bin_quan([row[-1] for row in testing_list]) #get true label from training

t=5 #change t here
stump=2 #change stump here
[ada_pred, vote, weights] = ada_boost(t, .001, training_list,stump)
fin_pred = ada_fin_pred(ada_pred, vote, len(training_list), t)

print('stump = ',stump,',training error at t =', t,' is ', total_error(true_label_bin, fin_pred))
print('stump = ',stump,',testing error at t =', t,' is ', total_error(true_label_bin_test, fin_pred))