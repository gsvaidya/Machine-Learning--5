
from matplotlib import pyplot as plt
import numpy as np
class Perceptron :

    """An implementation of the perceptron algorithm.
    Note that this implementation does not include a bias term"""

    def __init__(self, max_iterations=1000, learning_rate=0.2) :

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
            count = 0.0
            for i in range(len(X)) :
                if y[i] * self.discriminant(X[i])+self.b <= 0 :
                    self.w = self.w + y[i] * self.learning_rate * X[i] 
                    #self.b = self.b + y[i]
                    
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

        
    def test(self,X,y):
        count=0.0
        Eout=0.0
        for i in range(len(X)):
            if self.predict(X[i]) != y[i] :
                count = count + 1
        Eout = count/len(X)
        print 'eout is',Eout
        return Eout

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
        
        scores = np.dot(self.w, X)+self.b
        return np.sign(scores)

   
        
def generate_separable_data():
    xA,yA,xB,yB = [np.random.uniform(-1, 1) for i in range(4)]
    w = np.random.uniform(-1, 1, 2)
    print w,w.shape
    data=np.genfromtxt("heart.data", delimiter=",", comments="#")

    X = data[:,2:]
    #X_test = X[101:]
    #X = X[0:100]
    print 'X',X
    new_array = np.zeros(X.shape)
    a = -1
    b = 1
    minimum = np.amin(X)
    maximum = np.amax(X)
    for i in range(len(X)):
        for j in range(len(X[0])):   
            new_array[i] = (((b-a)*(X[i] - minimum))/maximum-minimum) + a
             
    
    print 'new_array is',new_array
    min2= np.amin(new_array)
    max2 = np.amax(new_array)
    print 'min2=',min2,'max2=',max2
    y = new_array[:,1]
    y_test = y[101:]
    print 'y',y
    y = y[0:100]
    
    #print X,X.shape
    X_test = X[101:]
    
    X = X[0:100]
    #print X_train,X_train.shape
    
    #print X_test,X_test.shape
    
    #print 'na is',new_array
    
    
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
    #p.learning_curve(100,X,y,X_test,y_test)
