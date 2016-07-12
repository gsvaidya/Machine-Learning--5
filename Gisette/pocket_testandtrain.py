
from matplotlib import pyplot as plt
import numpy as np
X = np.genfromtxt("gisette_train2.txt" , delimiter = None)
y = np.genfromtxt("gisette_labels.txt" , delimiter = " ", comments = "#")
class Perceptron :

    """An implementation of the perceptron algorithm.
    Note that this implementation does not include a bias term"""

    def __init__(self, max_iterations=100, learning_rate=0.2) :

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
        self.w = np.zeros(len(X[0]))

        self.b = 0
        #score = 0
        #score2 = 0
        w_pocket=0
        Ein2 = 1
        Ein = []
        converged = False
        iterations = 0
        count = 0.0
        Ein_final = []
        while (not converged and iterations < self.max_iterations) :
            converged = True
            for i in range(len(X)) :
                if y[i] * self.discriminant(X[i])+self.b <= 0 :
                    self.w = self.w + y[i] * self.learning_rate * X[i] 
                    self.b = self.b*self.learning_rate + y[i]
                    converged = False
                    

                    count = count + 1
##            Ein = count/len(X)
##            Ein_final.append(Ein)
##            Ein_fin = min(Ein_final)
            
            Ein = count/len(X)
            if (Ein<Ein2):
                w_pocket = self.w
            Ein2 = Ein
            
            #plot_data(X, y, self.w)
                #else:
                    #score = score + 1
                    #if (score > score2):
                        #wp = self.w
                        #score2 = score
    
                    
                            
                    
            iterations += 1
        self.converged = converged
        if converged :
            print w_pocket
            print Ein
            print 'converged in', iterations ,'iterations'
            
            #plot_data(X,y,w+self.b)
            
             

            #plot_data(X,y,self.w)
    def test(self,X,y):
        count=0.0
        Eout=0.0
        for i in range(len(X)):
            if y[i] * self.discriminant(X[i]) <= 0 :
                count = count + 1
            Eout = count/len(X)
        print Eout       


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
    X_test = np.genfromtxt("gisette_testing.txt" , delimiter = None)
    print X_test,X_test.shape
    X = np.genfromtxt("gisette_training.txt" , delimiter = None)
    print X,X.shape
    y_test = np.genfromtxt("labels_testing.txt" , delimiter = " ", comments = "#")
    y = np.genfromtxt("labels_training.txt" , delimiter = " ", comments = "#")
    return X,y,w,X_test,y_test
    
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
    X,y,w,X_test,y_test = generate_separable_data()
    p = Perceptron()
    generate_separable_data()             
    p.fit(X,y)
    p.test(X_test,y_test)
