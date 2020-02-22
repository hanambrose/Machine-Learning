import math
import statistics
from random import randrange, seed
#-----------------trainning data------------------------------------
with open('train.csv',mode='r') as f:
    training_list=[];
    for line in f:
        data_point=line.strip().split(',') 
        training_list.append(data_point)

num_set={0,5,9,11,12,13,14} # attribute's index with numeric value   
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
#####
#training_list= training_list[:20]                        
#--------------------testing data--------------------------------
with open('test.csv',mode='r') as f:
    testing_list=[];
    for line in f:
        data_point=line.strip().split(',') # 7*N matrix
        testing_list.append(data_point)

testing_list = str_2_flo(testing_list)
for i in obj:
    obj[i] = statistics.median([row[i] for row in testing_list])
    
for row in testing_list:
    for i in obj:
        if row[i] >= obj[i]:
            row[i] = 'yes'
        else:
            row[i] = 'no'
#####
#testing_list= testing_list[:20]          
#==============================================================================
Attr_dict ={'age':['yes','no'],
             'job':['admin.','unknown','unemployed','management','housemaid','entrepreneur','student','blue-collar','self-employed','retired','technician','services'],
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
    return obj

def attr_list(attr):
    obj={}
    for attr_val in attr:
        obj[attr_val]=0
    return obj
#----------------------------------------------------------------------------
def info_gain(branches, labels):
    total_size= float(sum([len(branches[attr_val]) for attr_val in branches])) 
    exp_ent = 0.0
    for attr_val in branches:
        size = float(len(branches[attr_val]))
        if size == 0:
            continue
        score = 0.0
        for class_val in labels:
            p = [row[-1] for row in branches[attr_val]].count(class_val) / size
            if p==0:
                temp=0
            else:
                temp=p*math.log2(1/p)
            score +=temp 
        exp_ent += score* (size / total_size)
    return exp_ent          
#----------------------------------------------------------------------------
def data_split(attr, dataset):
    branch_obj=branches_list(attr)
    for row in dataset:
        for attr_val in Attr_dict[attr]:
           if row[attr_index(attr)]==attr_val:
               branch_obj[attr_val].append(row)
    return branch_obj
#----------------------------------------------------------------------------
def best_spl(dataset):
    if dataset==[]:
        return 
    label_values = list(set(row[-1] for row in dataset)) 
    metric_obj = attr_list(Attr_dict)
    for attr in Attr_dict:
        branches = data_split(attr, dataset)
        metric_obj[attr] = info_gain(branches, label_values)# change metric here
    best_attr = min(metric_obj, key=metric_obj.get)
    best_branches = data_split(best_attr, dataset)  
    return {'best_attr':best_attr, 'best_branches':best_branches}
#----------------------------------------------------------------------------    
def leaf_node_label(group):
    majority_labels = [row[-1] for row in group]
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
    if not if_node_divisible(node['best_branches']):  
        
        for key in node['best_branches']:
            if  node['best_branches'][key]!= []:  
                 node[key] = leaf_node_label(node['best_branches'][key])
            else:
                node[key] = leaf_node_label(sum(node['best_branches'].values(),[])) 
        return
    if curr_depth >= max_depth:
        for key in node['best_branches']:
            if  node['best_branches'][key]!= []: 
                
                node[key] = leaf_node_label(node['best_branches'][key])   
            else:
                node[key] = leaf_node_label(sum(node['best_branches'].values(),[]))
        return  
    for key in node['best_branches']:
        if node['best_branches'][key]!= []: 
            node[key] = best_spl(node['best_branches'][key]) 
            internal_node(node[key], max_depth, curr_depth + 1)
        else:
            node[key] = leaf_node_label(sum(node['best_branches'].values(),[]))         
#----------------------------------------------------------------------------
def tree_build(train, max_depth):
	root = best_spl(train)
	internal_node(root, max_depth, 1)
	return root
#----------------------------------------------------------------------------
def label_predict(node, inst):
    if isinstance(node[inst[attr_index(node['best_attr'])]],dict):
        return label_predict(node[inst[attr_index(node['best_attr'])]],inst)
    else:
        return node[inst[attr_index(node['best_attr'])]]   
#===================================bagging starts here================================
#create a random subsample from the dataset with replacement
def subsample(dataset, ratio = 1.0):
    sample = list()
    n_sample = round(len(dataset)*ratio)
    while len(sample)< n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample 
   
#Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers)/ float(len(numbers))

#predict individual tree given certain number of trees
def bagging_predict(trees,row):
    predictions= [label_predict(tree,row) for tree in trees] 
    return max(set(predictions), key = predictions.count)

#randomly sample the dataset, make prediction on each sample tree then bag them together
def bagging(train, test, max_depth, sample_size, n_trees):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = tree_build(sample, max_depth)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test] #make prediction on test set
    return (predictions)
    
# Calculate error
def error_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] != predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
#===================================Report================================
 
seed(7)
max_depth = 13
sample_size = 0.20

for n_trees in [1,20,100]:
    predicted= bagging(training_list, testing_list, max_depth, sample_size, n_trees)
    actual = [row[-1] for row in training_list]
    print('Trees: %d' % n_trees)
    print('Testing Error: %.3f%%' % error_metric(actual, predicted))
