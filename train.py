import numpy as np
import pandas as pd

  
def tanhderiv(K):
 """
 used to calculate the derivative of tanh function.
 """
return 1- (np.tanh(K)**2)



def initialisetheta(m,n,a,yn):
 """
 used to randomly initialise the weights matrix and store it in the
 form of a list. Note that the bias term has been directly added to each weight matrix.
 """

theta=[]
    theta.append(np.random.randn(n+1,a[0]))
    t=len(a)
    for i in range(t-1):
        theta.append(np.random.randn(a[i]+1,a[i+1]))
    theta.append(np.random.randn(a[len(a)-1]+1,yn))
    return theta



def hiddenlayer(m,a):
"""
used to initialise the hidden layers with ones.
Bias term has been included here as well.

"""

hidden=[]
    for i in a:
        hidden.append(np.ones((m,i+1)))
    return hidden

def forwardpropogation(x,theta,hidden):
 """
 performs a pass of forward propogation and predicts the output
 of the pass.
 
 """
    hidden[0][:,1:]= np.tanh(np.dot(x,theta[0]))
    for i in range (len(hidden)-1):
        hidden[i+1][:,1:]= np.tanh(np.dot(hidden[i],theta[i+1]))
    output=np.tanh(np.dot(hidden[len(hidden)-1],theta[-1]))
    return (output,hidden)

def delta_function(m,hidden,output,y,theta):
    """
    performs a pass of back propogation and calculates the value
    of delta for each layer.
    
    """
    delta=[]
    delta.append(output-y)

    delta.insert(0, np.dot(delta[0], theta[len(hidden) ].T ))

    for i in range(len(hidden)-1):
        p =np.zeros((m,len(hidden[len(hidden)-i-1])))
        p= delta[0]* tanhderiv(hidden[len(hidden)-i-1])
        delta.insert(0, np.dot(p[:,1:],theta[len(hidden)-i-1].T))

    return delta

def update_theta(x,delta,hidden,theta):
    """
    the value of theta matrices in list "theta" are updated for each pass
    """
    
    print(np.size(x.T,0))
    theta[0]+= np.dot(x.T,delta[0][:,1:])
    for i in range(len(hidden)-1 ):
        theta[i+1]+=np.dot(hidden[i].T,delta[i+1][:,1:])
    theta[len(hidden)]+=np.dot(hidden[len(hidden)-1].T,delta[len(hidden)])
    return theta

def main():
    """
    sample run
    """
    x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]])  # Test data
    y = np.array([[0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1]]) #Test Labels
    
    m = np.size(x, 0)
    n = np.size(x, 1)
    yn = np.size(y, 1)
    
    a = [30,20,40] """ number of layers and nodes in each layer """
    X = np.ones((m,n+1))
    X[:,1:]=x
    output=np.zeros((m,4))
   """
   theta and hidden layer initialisation
   
   """

    theta = initialisetheta(m,n,a,yn)
    hidden = hiddenlayer(m,a)
    
   
  
   
    for i in range(10000):
        output,hidden= forwardpropogation(X,theta,hidden)
        delta=delta_function(m,hidden, output, y,theta)
        theta=update_theta(X,delta,hidden,theta)
    
    
    
main()
