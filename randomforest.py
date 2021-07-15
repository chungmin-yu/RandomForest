# -*- coding: utf-8 -*-
"""
Created on Sat May 29 23:09:55 2021

@author: user
"""

import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

num_size=0.3
num_depth=3
num_tree=1
random_select=0

''' 
num_size=float(input('Input #validation_data / #toal_ data (between 0 and 1): '))
num_depth=int(input('Input the depth of decision tree: '))
num_tree=int(input('Input the number of trees in random forest: '))
random_select=int(input('Extremely random forest? (Input 1 or 0 for Y/N): '))
'''
#wine dataset
wine = datasets.load_wine()
attr_length=len(wine.feature_names)
obj_length=len(wine.target_names)
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=num_size)

'''
# irirs dataset
iris = datasets.load_iris()
attr_length=len(iris.feature_names)
obj_length=len(iris.target_names)
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=num_size)
'''
#define node class, including getting derived target
class Node:
    def __init__(self, value=None, target=None):
        self.value = value
        self.target = target
        self.attribute = None
        self.threshold = None
        self.left_child = None 
        self.right_child = None 
    def get_obj(self):
        obj0=self.target.count(0)
        obj1=self.target.count(1)
        obj2=self.target.count(2)
        ma=max(obj0,obj1,obj2)
        if ma == obj0:
            return 0
        elif ma == obj1:
            return 1
        elif ma == obj2:
            return 2
        else:
            return -1

#calculate gini        
def cal_gini(left,right):
    total=len(left)+len(right)
    numL=len(left)
    numR=len(right)
    if numL==0 or numR==0:
        return 2
    giniL=1-pow((left.count(0)/numL), 2)-pow((left.count(1)/numL), 2)-pow((left.count(2)/numL), 2)
    giniR=1-pow((right.count(0)/numR), 2)-pow((right.count(1)/numR), 2)-pow((right.count(2)/numR), 2)
    return (numL/total)*giniL+(numR/total)*giniR

#create decision tree
def train(current, first_value, first_target, depth):
    global num_depth
    global random_select
    global attr_length
    
    #the condition of terminate of decision tree   
    if first_target.count(0)==0 and first_target.count(1)==0:
        return 
    elif first_target.count(1)==0 and first_target.count(2)==0:
        return
    elif first_target.count(2)==0 and first_target.count(0)==0:
        return
    elif depth == num_depth:
        return
    
    #initialize
    right_value=[]
    right_target=[]
    left_value=[]
    left_target=[]
    gini_min=2
    attr_min=-1
    thr_min=-1
    
    #attribute bagging
    if random_select:
        random_attr = random.randint(0,3)
        attr=[]
        thr=[]
        for i in range(len(first_value)):
            attr.append(first_value[i][random_attr])
        attr.sort()
        for i in range(len(attr)-1):
            temp=(attr[i]+attr[i+1])/2
            thr.append(temp)
        for th in thr:
            right_target.clear()
            left_target.clear()
            for i in range(len(first_value)):
                if first_value[i][random_attr] >= th:
                    right_target.append(first_target[i])
                else:
                    left_target.append(first_target[i])
            tmp=cal_gini(left_target, right_target)
            if tmp<gini_min:
                gini_min=tmp
                attr_min=random_attr
                thr_min=th
    else:
        for k in range(attr_length):
            attr=[]
            thr=[]
            for i in range(len(first_value)):
                attr.append(first_value[i][k])
            attr.sort()
            for i in range(len(attr)-1):
                temp=(attr[i]+attr[i+1])/2
                thr.append(temp)
            for th in thr:
                right_target.clear()
                left_target.clear()
                for i in range(len(first_value)):
                    if first_value[i][k] >= th:
                        right_target.append(first_target[i])
                    else:
                        left_target.append(first_target[i])
                tmp=cal_gini(left_target, right_target)
                if tmp<gini_min:
                    gini_min=tmp
                    attr_min=k
                    thr_min=th


    #After getting attribute and threshold, we partition left and right child to current node 
    right_target.clear()
    left_target.clear()
    current.attribute=attr_min
    current.threshold=thr_min
    for i in range(len(first_value)):
        if first_value[i][attr_min] >= thr_min:
            right_value.append(first_value[i])
            right_target.append(first_target[i])
        else:
            left_value.append(first_value[i])
            left_target.append(first_target[i])
            
    #recursive and assign left and right child to current node 
    if len(left_value) !=0:
        left_node=Node(left_value, left_target)
        current.left_child=left_node
        train(current.left_child, left_value, left_target, depth+1)
    if len(right_value) !=0:
        right_node=Node(right_value, right_target)
        current.right_child=right_node
        train(current.right_child, right_value, right_target, depth+1)
    
#check the validation data belong to which class through decision tree   
def validation(current_node, testdata):
    ans = -1
    while current_node.attribute != None:
        if testdata[current_node.attribute] >= current_node.threshold:
            ans = current_node.get_obj()
            current_node = current_node.right_child
        else:
            ans = current_node.get_obj()
            current_node = current_node.left_child
    ans = current_node.get_obj()    
    return ans

#decidion tree
tree_pred=[]
tree_acc=0
d_value = x_train.tolist()
d_target = y_train.tolist()
d = Node(d_value, d_target)
train(d, d_value, d_target,0)
d2_value = x_test.tolist()
d2_target = y_test.tolist()
for j in range(len(d2_value)):
    dd = validation(d,d2_value[j])
    tree_pred.append(dd)
    if dd == d2_target[j]:
        tree_acc+=1
print('DECISION TREE')
print('accuracy: ', tree_acc/len(d2_target))  
print('error rate: ', 1-tree_acc/len(d2_target))   
print('\nconfusion matrix:')
print(confusion_matrix(y_test, tree_pred)) 
print('\nclassification report:')
print(classification_report(y_test, tree_pred), '\n')

#initialize
forest=[]
a=0
p=[]
total=[]
acc=0
pred=[]
bag_pred=[]

bag=[]
num_bag=[]
for i in x_train.tolist():
    bag.append(i)
    num_bag.append([])


for i in range(num_tree):
    p.append([])
    total.append([])
    for j in range(obj_length):
        p[i].append(0)
        total[i].append(0)
        
        
#random forest(tree bagging)
for i in range(num_tree):
    seed = random.randint(0,100000)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train, y_train, test_size=0.3, random_state=seed)
    root_value = x_train2.tolist()
    root_target = y_train2.tolist()
    root = Node(root_value, root_target)
    train(root, root_value, root_target,0)
    forest.append(root)
    test_value = x_test2.tolist()
    test_target = y_test2.tolist()
    for j in range(len(test_value)):
        for k in range(len(x_train.tolist())):
            if test_value[j] == bag[k]:
                num_bag[k].append(i)
    for j in range(len(test_value)):
        v = validation(root,test_value[j])
        if v == test_target[j]:
            p[i][v]+=1
            total[i][v]+=1
        else:
            total[i][v]+=1           
    
    for j in range(obj_length): 
        if total[i][j] == 0:
            p[i][j]=0
        else:
            p[i][j]=p[i][j]/total[i][j]
        
      
#calculate out-of-bag error       
for i in range(len(x_train.tolist())):
    results=[]
    for k in num_bag[i]:
        res = validation(forest[k],x_train.tolist()[i])
        results.append(res)
    #better random forest prediction algorithm    
    weight_vote=[]
    for j in range(obj_length):
        weight_vote.append(0)
        for k in range(len(results)):
            if results[k] == j:
                weight_vote[j]+=p[k][j]
    vote=weight_vote.index(max(weight_vote))    
    #original random forest prediction algorithm
    #print(results)
    #vote=max(results)
    bag_pred.append(vote)
    if vote == y_train.tolist()[i]:
        a+=1    
       
#execute validation function and vote the result
v_value = x_test.tolist()
v_target = y_test.tolist()
for i in range(len(v_value)):
    results=[]
    for r in forest:
        res = validation(r,v_value[i])
        results.append(res)
    #better random forest prediction algorithm
    weight_vote=[]
    for j in range(obj_length):
        weight_vote.append(0)
        for k in range(len(results)):
            if results[k] == j:
                weight_vote[j]+=p[k][j]
    vote=weight_vote.index(max(weight_vote))
    #original random forest prediction algorithm
    #print(results)
    #vote=max(results)
    pred.append(vote)
    if vote == v_target[i]:
        acc+=1
 
#print result
print('RANDOM FOREST')
print('Accuracy: ', acc/len(v_target))   
print('Validation-set Error:', 1-acc/len(v_target))
print('Out-of-bag Error: ', 1-a/len(y_train.tolist())) 
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, pred))
print('\nClassification Report:')
print(classification_report(y_test, pred))
      
