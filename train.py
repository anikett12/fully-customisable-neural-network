import numpy as np
import pandas as pd


def tanhderiv(K):
    return 1- (np.tanh(K)**2)

def initialisetheta(m,n,a,yn):
    theta=[]
    theta.append(np.random.randn(n+1,a[0]))
    t=len(a)
    for i in range(t-1):
        theta.append(np.random.randn(a[i]+1,a[i+1]))
    theta.append(np.random.randn(a[len(a)-1]+1,yn))
    return theta

def hiddenlayer(m,a):
    hidden=[]
    for i in a:
        hidden.append(np.ones((m,i+1)))
    return hidden

def forwardpropogation(x,theta,hidden):

    hidden[0][:,1:]= np.tanh(np.dot(x,theta[0]))
    for i in range (len(hidden)-1):
        hidden[i+1][:,1:]= np.tanh(np.dot(hidden[i],theta[i+1]))
    output=np.tanh(np.dot(hidden[len(hidden)-1],theta[-1]))
    return (output,hidden)

def delta_function(m,hidden,output,y,theta):
    delta=[]
    delta.append(output-y)

    delta.insert(0, np.dot(delta[0], theta[len(hidden) ].T ))

    for i in range(len(hidden)-1):
        p =np.zeros((m,len(hidden[len(hidden)-i-1])))
        p= delta[0]* tanhderiv(hidden[len(hidden)-i-1])
        delta.insert(0, np.dot(p[:,1:],theta[len(hidden)-i-1].T))

    return delta

def update_theta(x,delta,hidden,theta):
    print(np.size(x.T,0))
    theta[0]+= np.dot(x.T,delta[0][:,1:])
    for i in range(len(hidden)-1 ):
        theta[i+1]+=np.dot(hidden[i].T,delta[i+1][:,1:])
    theta[len(hidden)]+=np.dot(hidden[len(hidden)-1].T,delta[len(hidden)])
    return theta

def main():
    df = np.loadtxt('corners.txt')


    x=df[:,0:2]

    m = np.size(x, 0)

    Y = df[:, 2:]
    u=df[:, 2:]
    u=u.astype(int)
    Y=Y.astype(int)

    y=np.zeros((4,m))
    for i in range(m):
        y[Y[i],i]=1
    y=y.T

    n = np.size(x, 1)
    yn = np.size(y, 1)
    count=0
    a = [30,20,40]
    X = np.ones((m,n+1))
    X[:,1:]=x
    output=np.zeros((m,4))
    theta = initialisetheta(m,n,a,yn)
    hidden = hiddenlayer(m,a)
    for i in range(10000):
        output,hidden= forwardpropogation(X,theta,hidden)
        delta=delta_function(m,hidden, output, y,theta)
        theta=update_theta(X,delta,hidden,theta)
    Z=np.argmax(output,axis=1)
    for i in range(m):
        if( u[i]==Z[i]):
           count=count+1
    print(count/m)
    print(count)

main()