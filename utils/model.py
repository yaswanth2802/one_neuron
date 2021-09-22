
import numpy as np
import pandas as pd


class Perceptron:
    def __init__(self,n,e):
        self.w=np.random.randn(3)
        self.e=e
        self.n=n
    def activationfunction(self,inputs,w):
        z=np.dot(inputs,w)
        return np.where(z>0,1,0)
    def fit(self,x,y):
        self.x=x
        self.y=y
        x_bias=np.c_[x,-np.ones((len(x),1))]
        
        for epoch in range(self.e):
            yhat=self.activationfunction(x_bias,self.w)
            self.error=y-yhat
            print(self.error)
            self.w=self.w+self.n*np.dot(x_bias.T,self.error)
    def predict(self,x):
        x_bias=np.c_[x,-np.ones((len(x),1))]
        return self.activationfunction(x_bias,self.w)
    
    def total_loss(self):
        
        total_loss = np.sum(self.error)
        print(f"total loss: {total_loss}")
        return total_loss
        