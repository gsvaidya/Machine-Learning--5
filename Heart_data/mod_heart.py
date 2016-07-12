import random
from matplotlib import pyplot as plt
import numpy as np
X = np.genfromtxt("gisette_train2.txt" , delimiter = None)
y = np.genfromtxt("gisette_labels.txt" , delimiter = " ", comments = "#")
class Perceptron :

    """An implementation of the perceptron algorithm.
    Note that this implementation does not include a bias term"""

    def __init__(self, max_iterations=200, learning_rate=0.2) :

        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

    def fit(self, X, y) :
        """
        Train a classifier using the perceptron training algorithm.
        After training the attribute 'w' will contain the perceptron weight vector.

        Parameters
        ----------

        X : ndarray, shape (num_examples, n_features)
        Training data.

        y : ndarray, shape (n_examples,)
        Array of labels.
        
        """
        self.w = np.random.uniform(-1,1,len(X[0]))
        c = 0.03
        l = np.zeros(len(X))
        converged = False
        iterations = 0
        Ein_final = []
        count = 0
        while (not converged and iterations < self.max_iterations) :
            converged = True
            for i in range(len(X)) :
                lamda = (y[i]*self.discriminant(X[i]))
                
                if (lamda < (c * np.linalg.norm(self.w))):
                    l[i] = lamda
                    converged = False
                    count = count + 1
            Ein = count/len(X)
           
            
            j = np.argmax(l)
            self.w = self.w + self.learning_rate*y[j]*X[j]
                    
                    #plot_data(X, y, self.w)
                            
                    
            iterations += 1
        print 'Ein',Ein
        self.converged = converged
        if converged :
            print 'converged in', iterations ,'iterations'
            
            #plot_data(X,y,wp+self.b)
            
             

            #plot_data(X,y,self.w)
    
    def test(self,X,y):
        print 'inside test'
        count = 0.0
        Eout = 0.0
        for i in range(len(X)):
            print 'inside for'
            if y[i] * self.discriminant(X[i]) <= 0 :
                count = count + 1
        Eout = count/len(X)
        print 'eout is',Eout
       
    def discriminant(self, x) :
        return np.dot(self.w, x)
           
    def predict(self, X) :
        """
        make predictions using a trained linear classifier

        Parameters
        ----------

        X : ndarray, shape (num_examples, n_features)
        Training data.
        """
        
        scores = np.dot(self.w, X)
        return np.sign(scores)

   
        
def generate_separable_data() :
    xA,yA,xB,yB = [np.random.uniform(-1, 1) for i in range(4)]
    w = np.random.uniform(-1, 1, 2)
    print w,w.shape
    data=np.genfromtxt("heart.data", delimiter=",", comments="#")
    
    print 'w is', w
    X = data[:,2:]
    #print X,X.shape
    X_test = X[101:]
    X = X[0:100]
    #print X_train,X_train.shape
    
    #print X_test,X_test.shape
    
    y = data[:,1]
    y_test = y[101:]
    y = y[0:100]
    
    return X_test,y_test,y,X,w
    
def plot_data(X, y, w) :
    fig = plt.figure(figsize=(5,5))
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    a = -w[0]/w[1]
    pts = np.linspace(-1,1)
    plt.plot(pts, a*pts, 'k-')
    cols = {1: 'r', -1: 'b'}
    for i in range(len(X)): 
        plt.plot(X[i][0], X[i][1], cols[y[i]]+'o')
    plt.show()

if __name__=='__main__' :
    X,y_test,y_test,y,w = generate_separable_data()
    p = Perceptron()
    generate_separable_data()             
    p.fit(X,y)
    p.test(X_test,y_test)
