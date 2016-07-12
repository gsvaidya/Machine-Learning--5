##
##
##
##
import numpy as np
from matplotlib import pyplot as plt
## 
##class Perceptron :
## 
##    """An implementation of the perceptron algorithm.
##    Note that this implementation does not include a bias term"""
## 
##    def __init__(self, max_iterations=6000, learning_rate=0.2) :
## 
##        self.max_iterations = max_iterations
##        self.learning_rate = learning_rate
## 
##    def fit(self, X, y) :
##        """
##        Train a classifier using the perceptron training algorithm.
##        After training the attribute 'w' will contain the perceptron weight vector.
## 
##        Parameters
##        ----------
## 
##        X : ndarray, shape (num_examples, n_features)
##        Training data.
## 
##        y : ndarray, shape (n_examples,)
##        Array of labels.
## 
##        """
##        print 'inside fit function'
##        self.w = np.zeros(len(X[0]))
##        converged = False
##        iterations = 0
##        count = 0.0
##        Ein_list = []
##        while (not converged and iterations < self.max_iterations) :
##            converged = True
##            
##            for i in range(len(X)) :
##                
##                if y[i] * self.discriminant(X[i]) <= 0 :
##                    self.w = self.w + y[i] * self.learning_rate * X[i]
##                    
##                    converged = False
##                    count = count + 1
##                    #print count
##                    #plot_data(X, y, self.w)
##            Ein = count/len(X)
##            Ein_list.append(Ein)
##            Ein = min(Ein_list)
##            
##            iterations += 1
##        print 'Ein',Ein
##        
##        self.converged = converged
##        if converged :
##            
##            print 'converged in %d iterations ' % iterations
##
##
##    def test(self,X,y):
##        
##        count=0.0
##        Eout=0.0
##        
##        for j in range(len(X)):
##            
##            if (y[j] * self.discriminant(X[j]) <= 0 ):
##                
##                count = count + 1
##                
##            Eout = count/len(X)
##        print 'Eout',Eout
##
##        
##    def discriminant(self, x) :
##        return np.dot(self.w, x)
## 
##    def predict(self, X) :
##        """
##        make predictions using a trained linear classifier
## 
##        Parameters
##        ----------
## 
##        X : ndarray, shape (num_examples, n_features)
##        Training data.
##        """
## 
##        scores = np.dot(self.w, X)
##        return np.sign(scores)
## 
def generate_separable_data() :
    xA,yA,xB,yB = [np.random.uniform(-1, 1) for i in range(4)]
    w = np.random.uniform(-1, 1, 2)
    print w,w.shape
    data=np.genfromtxt("heart.data", delimiter=",", comments="#")
    X = data[:,2:]
    mean = np.mean(X,0)
    std = np.std(X,0)
    print std
    std_matrix = np.zeros(X.shape)

    std_matrix = (X-mean)/(std)
    min = np.amin(std_matrix)
    print 'min' ,min
    max = np.amax(std_matrix)
    print 'max', max
    X_test = X[101:]

    X = X[0:100]
    y = data[:,1]
    y_test = y[101:]
    y = y[0:100]
    return X,y,w,X_test,y_test
 
##def plot_data(X, y, w) :
##    fig = plt.figure(figsize=(5,5))
##    plt.xlim(-1,1)
##    plt.ylim(-1,1)
##    a = -w[0]/w[1]
##    pts = np.linspace(-1,1)
##    plt.plot(pts, a*pts, 'k-')
##    cols = {1: 'r', -1: 'b'}
##    for i in range(len(X)): 
##        plt.plot(X[i][0], X[i][1], cols[y[i]]+'o')
##    plt.show()
## 
if __name__=='__main__' :
    X,y,w,X_test,y_test = generate_separable_data()
##    p = Perceptron()
##    p.fit(X,y)
##    p.test(X_test,y_test)