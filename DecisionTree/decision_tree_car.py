import math
with open('train.csv',mode='r') as f:
    training_list=[];
    for line in f:
        data_point=line.strip().split(',') 
        training_list.append(data_point)
        
Attr_dict={'buying': ['vhigh', 'high', 'med', 'low'], 
           'maint':  ['vhigh', 'high', 'med', 'low'],
           'doors':  ['2', '3', '4', '5more' ],
           'persons': ['2', '4', 'more'],
           'lug_boot':[ 'small', 'med', 'big'],
           'safety': ['low', 'med', 'high']}
#----------------------------------------------------------------------------
def attr_index(attr):
    attr_dict_int = {'buying': 0, 
                     'maint':1,
                     'doors':2,
                     'persons':3,
                     'lug_boot':4,
                     'safety':5}
    if attr not in attr_dict_int:
        attr_index = 0
    else:
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
def gini_index(branches, labels):
    total_size= float(sum([len(branches[attr_val]) for attr_val in branches])) 
    gini = 0.0
    for attr_val in branches:   
        size = float(len(branches[attr_val]))
        if size == 0:
            continue
        score = 0.0
        for class_val in labels: 
            p = [row[-1] for row in branches[attr_val]].count(class_val) /size
            score += p * p	       
        gini += (1.0 - score) * (size / total_size)
    return gini          

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
def m_error(branches, labels):
    total_size = float(sum([len(branches[attr_val]) for attr_val in branches]))
    m_err = 0.0
    for attr_val in branches:
        size = float(len(branches[attr_val]))
        if size == 0:
            continue
        score = 0.0
        temp=0
        for class_val in labels:
            p = [row[-1] for row in branches[attr_val]].count(class_val) / size
            temp=max(temp,p)
            score=1-temp
        m_err += score* (size / total_size)
    return m_err
#----------------------------------------------------------------------------
def data_split(attr, dataset):
    branches_obj=branches_list(attr)  
    for row in dataset:
        for attr_val in Attr_dict[attr]:
           if row[attr_index(attr)]==attr_val:
               branches_obj[attr_val].append(row)
    return branches_obj
#----------------------------------------------------------------------------
def best_spl(dataset):
    if dataset==[]:
        return 
    label_values = list(set(row[-1] for row in dataset)) 
    metric_obj = attr_list(Attr_dict)
    for attr in Attr_dict:
        branches = data_split(attr, dataset)
        metric_obj[attr] =  info_gain(branches, label_values)# change method gini, IG, ME
    best_attr = min(metric_obj, key=metric_obj.get)
    best_branches = data_split(best_attr, dataset)  

    return {'best_attr':best_attr, 'best_branches':best_branches}
#----------------------------------------------------------------------------    
# returns the majority label
def leaf_node_label(group):
    majority_labels = [row[-1] for row in group]
    return max(set(majority_labels), key=majority_labels.count)
#----------------------------------------------------------------------------
def if_node_divisible(branches_obj):
    non_empty_indices=[key for key in branches_obj if not (not branches_obj[key])]
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
#----------------------------------------------------------------------------     
def error(true_labels, predicted):
    count = 0  
    for i in range(len(true_labels)):
        if true_labels[i] != predicted[i]:
            count += 1
    return count / float(len(true_labels)) * 100.0
#----------------------------------------------------------------------------
with open('test.csv',mode='r') as f:
    testing_list=[];
    for line in f:
        data_point=line.strip().split(',') 
        testing_list.append(data_point)
#----------------------------------------------------------------------------
for depth in range(1,7):
    tree=tree_build(training_list, depth)
     
    true_labels = []
    pred_labels = []
    for row in training_list: #switch between training and testing for error
        true_labels.append(row[-1])
        pred = label_predict(tree, row)
        pred_labels.append(pred)
    
    print('training errors when depth = ', depth,'is',round(error(true_labels, pred_labels)),"%")
#----------------------------------------------------------------------------
for depth in range(1,7):
    tree=tree_build(training_list, depth)
     
    true_labels = []
    pred_labels = []
    for row in testing_list: #switch between training and testing for error
        true_labels.append(row[-1])
        pred = label_predict(tree, row)
        pred_labels.append(pred)
    
    print('testing errors when depth = ', depth,'is',round(error(true_labels, pred_labels)),"%")    
    
