#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay as CMD
from sklearn.metrics import classification_report


# In[2]:


data_df = pd.read_csv("data/cpp_df.csv")
partial_df = data_df.sample(frac = 0.001, random_state = 1).reset_index(drop=True)
X = partial_df.drop(["Antigen", "Kmerized_sequences"],axis=1)
y = partial_df.pop("Antigen")


# In[3]:


#Train test split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=123, stratify = y)

#Print sizes of the split data
print(f"X_train : {X_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"y_train : {y_train.shape}")
print(f"y_test : {y_test.shape}")


# In[4]:


#Check stratification ratios
y_train_strat = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
y_test_strat = len(y_test[y_test == 0]) / len(y_test[y_test == 1])
print(f"y_train : {y_train.value_counts()}")
print('Ratio of 0:1 in y_train: %0.2f' % y_train_strat )
print(f"y_test : {y_test.value_counts()}")
print('Ratio of 0:1 in y_test: %0.2f' % y_test_strat) 


# In[5]:


def summarize_results(grid_result, model_name):
    # summarize results
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%0.2f (%0.2f) with: %r" % (mean, stdev, param))
        
    # Best estimator object
    print("---------------------------------------------------")
    print(f"Best Model = {grid_result.best_estimator_}")
    print(f"Best Model parameters = {grid_result.best_params_}")

    #Score on the test set
    print("---------------------------------------------------")
    print("Accuracy score on the test set = %0.2f" % grid_result.score(X_test, y_test))
    print("---------------------------------------------------")
    
    #Summarizing and storing the grid search in a dataframe
    grid_search_results = pd.DataFrame(grid_result.cv_results_)
    grid_search_results = grid_search_results.sort_values(by = 'mean_test_score', ascending = False )
        
    #Output the info to a text file
    filename = "model outputs/"+model_name+"_grid_search_results.txt"
    f = open(filename, 'w')
    print(f"Best performing model = {grid_result.best_estimator_} \n", file = f)
    print("Accuracy score on the test set = %0.2f \n" % grid_result.score(X_test, y_test) , file = f)
    print(f"Model parameters = {grid_result.best_params_} \n", file = f)
    print(grid_search_results, file = f)
    f.close()


# In[6]:


def get_ROC_curve(grid_result, model_name, model):

    #calculate and plot receiver operating characteristics (ROC) and calculate area under the curve (AUC)
    if isinstance(model, RidgeClassifier): 
        dec_probs = grid_result.decision_function(X_test)
        y_proba = np.exp(dec_probs) / np.sum(np.exp(dec_probs))

    else:
        y_proba = grid_result.predict_proba(X_test)[:,1]
    
    
    fprs, tprs, thresholds = roc_curve(y_test, y_proba)  
    roc_auc = roc_auc_score(y_test, y_proba)    
    print(f'Area under curve (AUC) for {model_name} = %0.2f' % roc_auc)
    
    fig = plt.figure()
    plt.plot(fprs, tprs, color='darkorange',
             lw=2, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="best")

    #save ROC curve to file
    plt.savefig(f"model outputs/{model_name}_ROC_AUC_curve.pdf")
    plt.show()

   


# In[7]:


def get_confusion_matrix(grid_result, model_name):
        
    # Generate the confusion matrix

    # get best model from grid search
    best_model = grid_result.best_estimator_

    #fit best model on training data
    best_model_result = best_model.fit(X_train, y_train)

    #predict y_values by using the .predict method
    y_pred = best_model_result.predict(X_test)

    #generate confusion matrix
    my_confusion_matrix = confusion_matrix(y_test, y_pred)

            
    #Get classification report
    print(f"Classification report for {str(model_name)} \n")
    report_initial = classification_report(y_test, y_pred)
    print(report_initial)

    #save classification report to file
    filename = "model outputs/"+str(model_name)+"_classification_report.txt"
    f = open(filename, 'w')
    print(f"Classification report for {str(model_name)} \n", file=f)
    print(report_initial, file = f)
    f.close()
    
    # Saving the confusion matrix to file  

    #plot confusion matrix
    sns.heatmap(my_confusion_matrix, annot=True, fmt = '.0f') 
    plt.xlabel("Predicted classes")
    plt.ylabel("Actual classes")
    plt.title(f'confusion matrix for {model_name}')

    #save confusion matrix to file
    plt.savefig(f"model outputs/{model_name}_confusion_matrix.pdf")
    plt.show()


# In[8]:


def run_classifiers(model):
    
   #For use in various print statements later
    model_name = str(model).replace("()", "")
   
    #CREATE LISTS OF HYPER_PARAMETERS TO TUNE
    if isinstance(model, LogisticRegression):
        
        #Max_iter
        max_iter = [10000]
        
        #Create list of solvers
        solvers = ['newton-cg', 'lbfgs', 'liblinear']

        #Create list of penalty values
        penalty = ['l1', 'l2']

        #Create list of c_values
        c_values = [100, 10, 1.0, 0.1, 0.01]

        #Convert the lists to a dictionary called grid
        grid = dict(solver=solvers,penalty=penalty,C=c_values, max_iter=max_iter)
    
    if isinstance(model, RidgeClassifier):
        #Create list of alpha values
        alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        #Convert the lists to a dictionary called grid
        grid = dict(alpha=alpha)

    if isinstance(model, SVC):

        #Create list of kernels
        kernel = ['poly', 'rbf', 'sigmoid']

        #Create list of c_values
        C = [50, 10, 1.0, 0.1, 0.01]

        #Create list of gammas
        gamma = ['scale']
        
        #Probability
        probability = [True]

        #Convert the lists to a dictionary called grid
        grid = dict(kernel=kernel,C=C,gamma=gamma, probability=probability)
       
    if isinstance(model, RandomForestClassifier):
        
        #Create list of n_estimators
        n_estimators = [10, 100, 1000]

        #Create list of max_features
        max_features = ['sqrt', 'log2']

        #Convert the lists to a dictionary called grid
        grid = dict(n_estimators=n_estimators,max_features=max_features)
        
    if isinstance(model, GradientBoostingClassifier):
        
        #Create list of n_estimators
        n_estimators = [5, 15, 25, 35, 45]

        #Create list of learning rates
        learning_rate = [0.001, 0.01, 0.1]

        #Create list of max_depths
        max_depth = [3, 7, 9]

        #Convert the lists to a dictionary called grid
        grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
        
        
    if isinstance(model, MLPClassifier):  
        
        #Create list of solvers
        solvers = ['lbfgs', 'sgd', 'adam']

        #Create list of solvers
        max_iter = [100, 250, 500, 750, 1000, 1250, 1500]

        #Create list of alphas
        alpha = [0.1   , 0.01  , 0.001 , 0.0001]

        #Create list of hidden_layer_sizes
        hidden_layer_sizes = [10 , 20]

        #Convert the lists to a dictionary called grid
        grid = dict(solver=solvers,alpha=alpha,max_iter=max_iter, hidden_layer_sizes=hidden_layer_sizes)
        
    if isinstance(model, KNeighborsClassifier): 
        #Create list of n_neighbors
        n_neighbors = range(1, 21, 2)

        #Create list of weights
        weights = ['uniform', 'distance']

        #Create list of metrics
        metric = ['euclidean', 'manhattan', 'minkowski']

        #Convert the lists to a dictionary called grid
        grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
        
    if isinstance(model, DecisionTreeClassifier):

        #Create list of depths
        max_depth = range(1, 21, 2)

        #Create list of max_features
        max_features = ['auto', 'sqrt', 'log2', 'none']

        #Create list of min_samples_leaf
        min_samples_leaf = [1, 2, 3]

        #Create list of splitters
        splitter = ['best', 'random']

        #Create list of criterion
        criterion = ['gini', 'entropy']

        #Convert the lists to a dictionary called grid
        grid = dict(max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf, splitter=splitter, criterion=criterion)
    
    else:
        print(f"Model name {model} not found")

    #Define the grid search
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=5, verbose=1)

    #Fit the grid search
    grid_result = grid_search.fit(X_train, y_train)

    summarize_results(grid_result, model_name)
    get_ROC_curve(grid_result, model_name, model)
    get_confusion_matrix(grid_result, model_name)


# In[9]:


models_list = [LogisticRegression(), 
               RidgeClassifier(), 
               SVC(), 
               RandomForestClassifier(), 
               GradientBoostingClassifier(), 
               MLPClassifier(), 
               KNeighborsClassifier(), 
               DecisionTreeClassifier()]

for model in models_list:
    run_classifiers(model)


# In[ ]:




