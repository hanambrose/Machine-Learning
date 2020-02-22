import numpy as np 
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
#----------------------------------------------------------------------------
#This changed for random forest to randomly sample features
#----------------------------------------------------------------------------
#---------------------------------------------------------------------------- 
    
def best_spl(dataset, n_features):
    if dataset==[]:
        return 
    label_values = list(set(row[-1] for row in dataset)) 
    features = np.random.choice(list(Attr_dict), n_features, replace=False)
    metric_obj = attr_list(features)
    for attr in features:
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
def internal_node(node, max_depth, curr_depth, n_features):
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
            node[key] = best_spl(node['best_branches'][key],n_features) 
            internal_node(node[key], max_depth, curr_depth + 1, n_features)
        else:
            node[key] = leaf_node_label(sum(node['best_branches'].values(),[]))         
#----------------------------------------------------------------------------
def tree_build(train, max_depth, n_features):
	root = best_spl(train,n_features)
	internal_node(root, max_depth, 1, n_features)
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
    return max(set(predictions), key = predictions.count)#selecting the most common prediction made by the bagged tree

#===================================Random Forest starts here================================   
# Random Forest Algorithm
def random_forest(train, test, max_depth, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = tree_build(sample, max_depth, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return(predictions)
    
# Calculate error
def error_metric(actual, predicted):
	mistake = 0
	for i in range(len(actual)):
		if actual[i] != predicted[i]:
			mistake += 1
	return mistake / float(len(actual)) * 100.0

#===================================Report================================
seed(1)
max_depth = 13
sample_size = 0.20

for n_features in [2,4,6]:
  print('Number of features', n_features)
  for n_trees in [1,10,100]:
      predicted= random_forest(training_list, testing_list, max_depth, sample_size, n_trees, n_features)
      actual = [row[-1] for row in testing_list]
      print('Trees: %d' % n_trees)
      print('Testing Error: %.3f%%' % error_metric(actual, predicted))