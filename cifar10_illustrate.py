import pickle # To load data from CIPHAR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial # To create nn functions with defined K's
from random import random,  sample
from sklearn.decomposition import PCA # To visualize the data and see feature distinction 
from sklearn.manifold import MDS # To visualize the data and see feature distinction 
from tqdm import tqdm # Measure time taken / esitmated time 
from numba import jit # For multithreading / code speed up

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

# PCA graphing to try and visualize any boundary between X and Y
def nd_image_2d(X, Y, f=PCA, randomize=0):
    n = 2
    X_ = pd.DataFrame(X)
    Y_ = pd.DataFrame(data=Y, columns=['label'])
    if randomize>0:
        indx = sample(range(X.shape[0]), randomize)
        X_ = X_.loc[indx].reset_index()
        Y_ = Y_.loc[indx].reset_index()

    pca = f(n_components = n)
    pc = pca.fit_transform(X_)
    
    df = pd.concat([pd.DataFrame(data=pc, columns=['pc_'+str(i+1) for i in range(n)]), Y_], axis = 1).loc[sample(range(len(X_)), 500)].reset_index()
    
    fig = plt.figure()
    ax = fig.add_subplot()
    groups = df.groupby('label')
    for name, group in groups:
        ax.scatter(group.pc_1, group.pc_2, marker = 'o', label=name)
    ax.legend()
    return df

def nd_image_3d(X, Y, f=PCA, randomize=0):
    n = 3
    X_ = pd.DataFrame(X)
    Y_ = pd.DataFrame(data=Y, columns=['label'])

    if randomize > 0:
        indx = sample(range(X.shape[0]), randomize)
        X_ = X_.loc[indx].reset_index()
        Y_ = Y_.loc[indx].reset_index()
        
    pca = f(n_components = n)
    pc = pca.fit_transform(X_)
    
    df = pd.concat([pd.DataFrame(data=pc, columns=['pc_'+str(i+1) for i in range(n)]), Y_], axis = 1).loc[sample(range(len(X_)), 500)].reset_index()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    groups = df.groupby('label')
    for name, group in groups:
        ax.scatter(group.pc_1, group.pc_2, group.pc_3, marker = 'o', label=name)
    ax.legend()
    return df
    
def class_acc(pred, gt, is_int=False):
    """Returns the accuracy between categorical lists"""
    acc_mat = [i==j for i,j in zip(pred, gt)]
    if is_int:
        return round(np.mean(acc_mat)*100,2)
    return f'{round(np.mean(acc_mat)*100,2)}%'

def cifar10_classifier_random(x):
    """Returns a random class between 1 and 9 for any input"""
    return int(random()*10)

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
    # How many datasets to load (5 would load all 50,000 images)
    batches = 5
    datadict = [None]*batches
    
    # Load each file over a loop
    for i in range(batches):
        datadict[i] = unpickle('cifar-10-batches-py/data_batch_'+str(i+1))
    datadic_test = unpickle('cifar-10-batches-py/test_batch')
    #datadict = unpickle('/home/kamarain/Data/cifar-10-batches-py/test_batch')
    
    # Concatenate the dataset into training and test sets
    X = np.concatenate([datadict[i]["data"] for i in range(batches)])
    Y = np.concatenate([datadict[i]["labels"] for i in range(batches)])
    X_test = datadic_test["data"]
    Y_test = datadic_test["labels"] 
    
    print(X.shape)
    
    # Reshape the dataset to be able to view the images
    labeldict = unpickle('cifar-10-batches-py/batches.meta')
    label_names = labeldict["label_names"]
    X = X.reshape(batches*10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8").astype('int')
    Y = np.array(Y)
    
    X_test = X_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8").astype('int')
    Y_test = np.array(Y_test)
 
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
    X = X.reshape(batches*10000, 3*32*32)
    X_test = X_test.reshape(10000, 3*32*32)
    
    nd_image_2d(X, Y, PCA)
    # nd_image_2d(X, Y, MDS)
    
    # Check if the accuracy function works
    print(f'gt vs gt = {class_acc(Y,Y)}\n')
    
    # Try a random classifier 
    Y_pred = [cifar10_classifier_random(X[i]) for i in range(X.shape[0])]
    print(f'random vs gt = {class_acc(Y_pred, Y)}\n')
    
    # Try my own 1-NN Classifier
    Y_pred = [cifar10_classifier_1nn(X_test[x], X, Y) for x in tqdm(range(100))]
    print(f'\nEuclidean Distance pred vs gt = {class_acc(Y_pred, Y_test)}\n')
    Y_pred = [cifar10_classifier_1nn(X_test[x], X, Y, f=manhattan_matrix_dist_opt) for x in tqdm(range(100))]
    print(f'\nManhattan Distance pred vs gt = {class_acc(Y_pred, Y_test)}\n')

    # Try my own 3-NN Classifier
    Y_pred = [cifar10_classifier_knn(X_test[x], X, Y, k=3) for x in tqdm(range(100))]
    print(f'\n3NN Euclidean Distance pred vs gt = {class_acc(Y_pred, Y_test)}\n')
    Y_pred = [cifar10_classifier_knn(X_test[x], X, Y, f=manhattan_matrix_dist_opt, k=3) for x in tqdm(range(100))]
    print(f'\n3NN Manhattan Distance pred vs gt = {class_acc(Y_pred, Y_test)}\n')
    
    # Plot the k-value vs the accuracy of the algo over multiple distance functions
    k_val = list(range(1,10))
    acc = [0]*len(k_val)
    acc_m = [0]*len(k_val)
    for k in k_val:
        acc[k-1] = class_acc([cifar10_classifier_knn(X_test[i], X, Y, k=k) for i in tqdm(range(1000))], Y_test, is_int=True)
        acc_m[k-1] = class_acc([cifar10_classifier_knn(X_test[i], X, Y, f=manhattan_matrix_dist_opt, k=k) for i in tqdm(range(1000))], Y_test, is_int=True)
    
    plt.figure(1)
    plt.clf()
    plt.plot(range(1,len(acc)+1), acc, label='euclidean')
    plt.plot(range(1,len(acc_m)+1), acc_m, label='manhattan')
    plt.legend()
    plt.show()
    
   

