
from matplotlib import pyplot 
import numpy as np
X = np.genfromtxt("gisette_train2.txt" , delimiter = None)
y = np.genfromtxt("gisette_labels.txt" , delimiter = " ", comments = "#")
class Perceptron :

    """An implementation of the perceptron algorithm.
    Note that this implementation does not include a bias term"""

    def __init__(self, max_iterations=1000, learning_rate=0.2) :

        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

    def fit(self, X, y) :
       
        self.w = np.zeros(len(X[0]))
        self.b = 0
        converged = False
        iterations = 0
        
        count = 0
        while (not converged and iterations < self.max_iterations) :
            converged = True
            for i in range(len(X)) :
                if y[i] * self.discriminant(X[i]) <= 0 :
                    self.w = self.w + y[i] * self.learning_rate * X[i] 
                    self.b = self.b + y[i]
                    
                    converged = False
                    count = count + 1
            Ein = count/len(X)
                
                    #plot_data(X, y, self.w)
            iterations += 1
        self.converged = converged
        if converged :
            print 'converged in', iterations ,'iterations'
            print 'Ein=', Ein
            #plot_data(X,y,self.w+self.b)
            
             

            #plot_data(X,y,self.w)

    def learning_curve(self,X_train,y_train,X_test,y_test):
        Eout = np.zeros(5)
        Accuracy = np.zeros(5)
        examples = np.zeros(5)
        j = 0
        for i in range(0,5):
            examples[j]=4**i
            X_curve = X_train[0:4**i,:]
            y_curve = y_train[0:4**i]

            self.fit(X_curve,y_curve)
            Eout[j] = self.test(X_test,y_test)
            Accuracy[j] = 100 - 100*(Eout[j])
            j = j + 1

        pyplot.plot(examples,Accuracy)
        
        pyplot.xscale('log',basex=4)
        pyplot.xlabel('number of examples',fontsize=18)
        pyplot.ylabel('Accuracy',fontsize=18)
        
        pyplot.show()

    def test(self,X,y):
        count=0.0
        Eout=0.0
        for i in range(len(X)):
            if y[i] * self.discriminant(X[i]) <= 0 :
                count = count + 1
        Eout = count/len(X)
        print Eout
        return Eout


    def discriminant(self, x) :
        dummy = np.dot(self.w, x)
        return dummy + self.b
            
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

   
        
def generate_separable_data():
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
    p.learning_curve(X,y,X_test,y_test)