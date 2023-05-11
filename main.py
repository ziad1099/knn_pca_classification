import numpy as np
import matplotlib.pyplot as plt
import csv
import random
#define knn and pca funcations
def knn_algo(k,x_train,y_train,x_test):
    y_pred=[]
    for i in range(len(x_test)):
        distances=[]
        for j in range(len(x_train)):
            distance=sum([abs(x_test[i][n] - x_train[j][n]) for n in range(len(x_test[i]))])
            distances.append((distance,y_train[j]))
        distances.sort()
        k_nearest_neighbors = distances[:k]
        k_labels=[neighbors[1] for neighbors in  k_nearest_neighbors]
        y_pred.append(max(set(k_labels),key=k_labels.count))
    return y_pred

def pca(X, n_components):
    # Calculate the mean of each column
    mean = np.mean(X, axis=0)

    # Subtract the mean from each element in X
    X = X - mean

    # Calculate the covariance matrix of X
    cov = np.cov(X.T)

    # Calculate the eigenvectors and eigenvalues of the covariance matrix
    eigvals, eigvecs = np.linalg.eig(cov)

    # Sort the eigenvectors in descending order by their corresponding eigenvalues
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:,idx]

    # Select the top n eigenvectors to use as the principal components
    components = eigvecs[:,:n_components]
    

    # Project the data onto the new subspace
    projected = X.dot(components)

    return projected

# function to split dataset
def test_train_split(array1,array2,test_size,random_state):
    random.seed(random_state)
    x_train=[]
    x_test=[]
    y_train=[]
    y_test=[]
    length=len(array2)
    train_size=length-(length*test_size)
    r=random.sample(range(0,length),length)
    for i in range(length):
        if i<train_size:
            x_train.append(array1[r[i]])
            y_train.append(array2[r[i]])
        else:
            x_test.append(array1[r[i]])
            y_test.append(array2[r[i]])
    
    return np.array(x_train),np.array(x_test),np.array(y_train),np.array(y_test)

# accuracy score funcation
def accuracy_score(array1,array2):
    score=0
    for i in range(len(array1)):
        if array1[i]==array2[i]:
            score+=1
    return score/len(array1)

# import dataset function
def import_dataset(file_name):
    file=open(file_name,'r')
    x=list(csv.reader(file,delimiter=","))
    data=[]
    c=[]
    for i in x:
        p=[float(i[n]) for n in range(len(i))]
        data.append(p[0:-1])
        c.append(p[-1])
    return np.array(data),np.array(c)

# Main 
if __name__=="__main__":
    # import dataset
    X,y=import_dataset("seeds.csv")

    # split datset to train and test
    X_train, X_test, y_train, y_test = test_train_split(
                X, y, test_size = 0.2, random_state=27)

    print(f"before FR:\n {X_train[1]} ==>from train\n {X_test[1]} ==>from test")
    # feature reduction by pca
    X_train_pca=pca(X_train,5)
    X_test_pca=pca(X_test,5)
    print(f"after FR:\n {X_train_pca[1]} ==>from train\n {X_test_pca[1]} ==>from test\n")
    acc=[]
    k=list(range(1,20))
    for i in k:
        # classification by knn
        y_pred=knn_algo(i,X_train_pca, y_train,X_test_pca)
        acc.append(accuracy_score(y_test,y_pred))
    print(f"Max accuracy is {max(acc)} when K= {acc.index(max(acc))+1}\n")
    plt.xlabel('K neighbers')
    plt.ylabel('Accuracy')
    plt.bar(k,acc)
    plt.show()
#     Ziad Helaly
