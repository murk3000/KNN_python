import sys
import os

sys.path.append(os.path.abspath('D:\\Grad\\Period 1\\Intro to PR and ML'))
from muneeb_libs import cifar10

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random
from functools import partial # To create nn functions with defined K's
from tqdm import tqdm # Measure time taken / esitmated time 
from numba import jit # For multithreading / code speed up
from sklearn.decomposition import PCA # To visualize the data and see feature distinction 
from sklearn.manifold import MDS # To visualize the data and see feature distinction

@jit(nopython=True, fastmath=True)
def simple_matrix_dist_opt(x,y):
    """Returns the Euclidean distance (square root of sum of squares of each element) of two arrays"""
    return np.array([np.sum((x[i]-y[i])**2) for i in range(x.shape[0])]) #  Taking the squareroot or not won't change the ordering of the knn

def manhattan_matrix_dist_opt(x,y):
    """Returns the Manhattan distance (sum of absolute value of each element) of two arrays"""
    delta = np.sum((np.abs(x-y)), axis=1).transpose()
    return(delta)

def rgb_matrix_dist(x,y):
    """Returns the Average Euclidean distance over RGB channels"""
    delta = np.square(x-y)
    delta = np.mean(np.sqrt(np.sum(delta.reshape(x.shape[0], 32*32, 3).transpose(2,0,1), axis=2)), axis=0)
    return delta

def cifar10_classifier_knn(x, trdata, trlabels, f = simple_matrix_dist_opt, k=1):
    """
    Returns the nearest neighbour label of x from trdata

    Parameters
    ----------
    x : object
        Any single object to compute knn against
    trdata : list
        list of objects that the object will be compared to 
    trlabels : list
        list of labels of trdata
    f : function, optional
        The function to use when comparing x and each element of trdata. The default is Eculidean Distance.
    k : int, optional
        The number of nearest neighbours to compute and then vote for the label. The default is 1.

    Returns
    -------
    int
        The label of the nearest neighbour with given specifications.

    """
    
    # Sets x to be of the same size as trdata to do a single array subtraction 
    x_array = np.repeat([x], trdata.shape[0], axis = 0) 
    dist = f(trdata, x_array)
    
    # Find the most votes/least distance combination based on k 
    df = pd.DataFrame(dist, columns=['distance']).reset_index().sort_values(by='distance', ascending=True)[:k]
    df['index'] = [trlabels[i] for i in df['index']]
    df = df.groupby('index').agg(['count', 'mean']).droplevel(0, axis=1).sort_values(by=['count', 'mean'], ascending=[False, True])    
    
    # return df
    return df[:1].index

# A partial function on top of the knn function to create the 1nn function
cifar10_classifier_1nn = partial(cifar10_classifier_knn, k=1)

if __name__ == '__main__':

    # Load dataset (1 would load 10,000 images, 5 would load all 50,000 images)
    batches = 5
    X,Y, X_test, Y_test, labeldict, label_names = cifar10.load_cifar10(batches)
 
    # Randomly show some images   
    for i in range(X.shape[0]):
        # Show some images randomly
        if random() > 0.99999:
            plt.figure(1);
            plt.clf()
            plt.imshow(X[i])
            plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
            plt.pause(1)
    
    # Undo the reshape to pass a dataset of vectors in the KNN algorithm
    X = X.reshape(batches*10000, 3*32*32).astype('int')
    X_test = X_test.reshape(10000, 3*32*32).astype('int')
    
    cifar10.nd_image_2d(X, Y, PCA)
    # nd_image_2d(X, Y, MDS)
    
    # Check if the accuracy function works
    print(f'gt vs gt = {cifar10.class_acc(Y,Y)}\n')
    
    # Try a random classifier 
    Y_pred = [cifar10.cifar10_classifier_random(X[i]) for i in range(X.shape[0])]
    print(f'random vs gt = {cifar10.class_acc(Y_pred, Y)}\n')
    
    # Try my own 1-NN Classifier
    Y_pred = [cifar10_classifier_1nn(X_test[x], X, Y) for x in tqdm(range(100))]
    print(f'\nEuclidean Distance pred vs gt = {cifar10.class_acc(Y_pred, Y_test)}\n')
    Y_pred = [cifar10_classifier_1nn(X_test[x], X, Y, f=manhattan_matrix_dist_opt) for x in tqdm(range(100))]
    print(f'\nManhattan Distance pred vs gt = {cifar10.class_acc(Y_pred, Y_test)}\n')

    # Try my own 3-NN Classifier
    Y_pred = [cifar10_classifier_knn(X_test[x], X, Y, k=3) for x in tqdm(range(100))]
    print(f'\n3NN Euclidean Distance pred vs gt = {cifar10.class_acc(Y_pred, Y_test)}\n')
    Y_pred = [cifar10_classifier_knn(X_test[x], X, Y, f=manhattan_matrix_dist_opt, k=3) for x in tqdm(range(100))]
    print(f'\n3NN Manhattan Distance pred vs gt = {cifar10.class_acc(Y_pred, Y_test)}\n')
    
    # Plot the k-value vs the accuracy of the algo over multiple distance functions
    k_val = list(range(1,10))
    acc = [0]*len(k_val)
    acc_m = [0]*len(k_val)
    for k in k_val:
        acc[k-1] = cifar10.class_acc([cifar10_classifier_knn(X_test[i], X, Y, k=k) for i in tqdm(range(1000))], Y_test, is_int=True)
        acc_m[k-1] = cifar10.class_acc([cifar10_classifier_knn(X_test[i], X, Y, f=manhattan_matrix_dist_opt, k=k) for i in tqdm(range(1000))], Y_test, is_int=True)
    
    plt.figure(1)
    plt.clf()
    plt.plot(range(1,len(acc)+1), acc, label='euclidean')
    plt.plot(range(1,len(acc_m)+1), acc_m, label='manhattan')
    plt.legend()
    plt.show()
    
   

