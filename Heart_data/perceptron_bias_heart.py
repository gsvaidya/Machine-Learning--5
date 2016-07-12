
from matplotlib import pyplot as plt
import numpy as np
class Perceptron :

    """An implementation of the perceptron algorithm.
    Note that this implementation does not include a bias term"""

    def __init__(self, max_iterations=5000, learning_rate=0.2) :

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
        converged = False
        iterations = 0
        dummy = np.zeros
        count = 0.0
        Ein_list=[]
        while (not converged and iterations < self.max_iterations) :
            converged = True
            for i in range(len(X)) :
                if y[i] * self.discriminant(X[i]) <= 0 :
                    self.w = self.w + y[i] * self.learning_rate * X[i]+self.b 
                    self.b = self.b*self.learning_rate + y[i]
                    
                    converged = False
                    count = count + 1
            Ein = count/len(X)
            Ein_list.append(Ein)
            Ein = min(Ein_list)
            
                    #plot_data(X, y, self.w)
            iterations += 1
        print 'Ein', Ein
        self.converged = converged
        if converged :
            print 'converged in', iterations ,'iterations'
            
            #plot_data(X,y,self.w+self.b)
            
             

            #plot_data(X,y,self.w)
####    def learning_curve(self,max_iterations,X_train,y_train,X_test,y_test):
####        self.max_iterations = max_iterations
####        Eout = np.zeros(self.max_iterations)
####        Accuracy = np.zeros(self.max_iterations)
####        examples = np.zeros(self.max_iterations)
####        j=0
####        while (j < (self.max_iterations)):
####            examples[j]=50+10*j
####            #X_curve = X_train[0:4**i,:]
####            #y_curve = y_train[0:4**i]
####
####            self.fit(X_train,y_train)
####            Eout[j] = self.test(X_test,y_test)
####            
####            Accuracy[j] = 100 - 100*(Eout[j])
####            j = j + 1
####        
####        plt.plot(examples,Accuracy)
####        
####        
####        plt.show()

    def test(self,X,y):
        count=0.0
        Eout=0.0
        for i in range(len(X)):
            if y[i]*self.discriminant(X[i])<=0 :
                count = count + 1
        Eout = count/len(X)
        print 'eout is',Eout


    def discriminant(self, x) :
        return np.dot(self.w, x)
        
            
    def predict(self, X) :
        
        scores = np.dot(self.w, X)
        return np.sign(scores)+self.b

   
        
def generate_separable_data():
    xA,yA,xB,yB = [np.random.uniform(-1, 1) for i in range(4)]
    w = np.random.uniform(-1, 1, 2)
    print w,w.shape
    data=np.genfromtxt("heart.data", delimiter=",", comments="#")

    X = data[:,2:]
    #X_test = X[101:]
    #X = X[0:100]

    
    y = data[:,1]
    y_test = y[170:]
    
    y = y[0:170]
    
    
    #print X,X.shape
    X_test = X[170:]
    
    X = X[0:170]
    #print X_train,X_train.shape
    
    #print X_test,X_test.shape
    
    
    
    
    #print y,y_test
    #y = np.sign(np.dot(X, w))
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
    p.learning_curve(1000,X,y,X_test,y_test)
