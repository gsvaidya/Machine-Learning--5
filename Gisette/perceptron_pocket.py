
from matplotlib import pyplot as plt
import numpy as np
#X = np.genfromtxt("gisette_train2.txt" , delimiter = None)
#y = np.genfromtxt("gisette_labels.txt" , delimiter = " ", comments = "#")
class Perceptron :

    """An implementation of the perceptron algorithm.
    Note that this implementation does not include a bias term"""

    def __init__(self, max_iterations=100, learning_rate=0.2) :

        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

    def fit(self, X, y) :
        self.w = np.zeros(len(X[0]))

        self.b = 0
        w_pocket=0
        Ein2 = 1
        converged = False
        iterations = 0
        count = 0.0
        while (not converged and iterations < self.max_iterations) :
            converged = True
            for i in range(len(X)) :
                if y[i] * self.discriminant(X[i]) <= 0 :
                    self.w = self.w + y[i] * self.learning_rate * X[i] + self.b
                    self.b = self.b + y[i]*self.learning_rate
                    converged = False
                    count = count + 1
            Ein = count/len(X)
            if (Ein<Ein2):
                w_pocket = self.w
            Ein2 = Ein
                            
                    
            iterations += 1
        self.converged = converged
        if converged :
            print w_pocket
            print Ein
            print 'converged in', iterations ,'iterations'
            
            #plot_data(X,y,w+self.b)
            
             

            #plot_data(X,y,self.w)
            
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
        return np.sign(scores)+self.b

   
        
def generate_separable_data(N) :
    xA,yA,xB,yB = [np.random.uniform(-1, 1) for i in range(4)]
    w = np.random.uniform(-1, 1, 2)
    print w,w.shape
    X = np.genfromtxt("gisette_train2.txt" , delimiter = None)
    print X,X.shape
    y = np.genfromtxt("gisette_labels.txt" , delimiter = " ", comments = "#")
    return X,y,w
    
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
    X,y,w = generate_separable_data(40)
    p = Perceptron()
    p.fit(X,y)
