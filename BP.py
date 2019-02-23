"""
Created on Mon Feb 11 10:10:43 2019

@author: HARI
"""

import numpy  as np
import random

class neural_network():
    def init(self,ni,nh):
        
        for i in range(0,ni):
            
            if(i!=0):
                self.n_input[i]=0
            else:
                self.n_input[0]=1
        print("initial input matrix is",end="")
        for i in range (1,ni+1):
            print(n_input[i])

        print("initial weights are")
        for i  in range(0,ni):
            print("")
            for j in range(0,nh):
                self.weights[i][j]=0
                print(self.weights[i][j],end="")

        return self.n_input,self.weights

    def calc_hidout(self,n_input,w):
        print("Z for hidden layer is [ ",end="")
        for i in range(0,len(w)):
            for j in range(0,len(w[0])):
                self.z[i]=0
                self.z[i]+=self.w[i][j]*self.n_input[j]
            print(self.z[i],end="")
        print("]")
        return self.z
        
    # Now for our function definition. Sigmoid.
    def sigmoid(self,x):
 
        self.ans=1 / (1 + math.exp(-x))
        return self.ans

    # Sigmoid function derivative.
    def dsigmoid(self,y):
        self.dans=y * (1 - y)
        return self.dans
  
    def calc_output(self,a,wp):
        print("a for output layer is [",end="")
        for i in range(0,len(wp)):
            for j in range(0,len(wp[0])):
                self.t[i]=0
                self.t[i]+=self.wp[i][j]*self.a[j]
            print(self.t[i],end="")
        print("]")
        return self.t

    def update_H2O(self,wp,t,y,z,lmda):
        for i in range(0,len(wp)):
            for j in range(0,len(t)):
               self.wp[i][j]-=lmda*(-1)*(t[j]-y[j]*(1-y[j]*z[i]))
        return self.wp

    def update_I2H(self,output_errors,wp,w,x,z,lmda):
        for i in range(0,len(wp)):
            self.hidden_errors[i]=0
            for j in range(0,1):
                self.hidden_errors[i]+=wp[i][j]*output_errors
        

        for i in range(0,len(w)):
            for j in range(0,len(self.hidden_errors)):
                self.delta=(-1)*self.hidden_errors[j]*self.z[j]*(1-z[j])*x[i]
                self.w[i][j]+=self.lmda*self.delta
        
        print("Delta=",self.delta)
        return self.w


    def train(w,wp,lmda,outputs,inputs,hidden,examples):
        for i in range(0,len(examples)):
            x[0]=1
            x[1]=random.uniform(-50,50)
            x[2]=random.uniform(-50,50)
