# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 10:10:43 2019

@author: HARI
"""

import numpy  as np
import random
import math

class NN():

    def init(self,ni,nh):
        ############################################################ 
        #w is randomly defined
        print("initial weights are")
        for i  in range(0,nh):
            print("")
            for j in range(0,ni+1):
                self.w[i][j]=random.uniform(-1,1)
                print(self.w[i][j],end="")

        #############################################################
        #wp is randomly definned
        print("final weights are")
        for i  in range(0,nh+1):
            print("")
            self.wp[i]=random.uniform(-1,1)
            print(self.wp[i],end="")
        print("")

        ###########################################################
        return self.w,self.wp


    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def rand_input(self,ni):
        #input x is randomly defined
        self.x=()
        for i in range(0,ni+1):
            if(i!=0):
                self.x[i]=random.uniform(-10,10)
            else:
                self.x[0]=1
        
        print("input matrix is",end="")
        for i in range (1,ni+1):
            print("input value",i,":",x[i])
        ##############################################################
        return self.x
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




    def calc_hidout(self,x,w):
        z=()
        print("Z for hidden layer is [ ",end="")
        for i in range(0,len(w)):
            for j in range(0,len(w[0])):
                self.z[i]=0
                self.z[i]+=self.w[i][j]*self.x[j]
            print(self.z[i],end="")
        print("]")

        #adding bias as a[0]=1
        self.a=()
        self.a[0]=1

        for i in range (1,len(self.z+1)):
            self.a[i]=1 / (1 + math.exp(-z[i]))
    
        return self.z,self.a

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 


    # Now for our function definition. Sigmoid.
    def sigmoid(self,x):
 
        self.ans=1 / (1 + math.exp(-x))
        return self.ans
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    # Sigmoid function derivative.
    def dsigmoid(self,y):
        self.dans=y * (1 - y)
        return self.dans
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def calc_output(self,a,wp):
        print("output layer is [",end="")
        for i in range(0,len(wp)):

            self.zp+=self.wp[i]*self.a[i]

            
        self.y=1 / (1 + math.exp(-self.zp))
        print(self.y,"]")

        return self.zp,self.y


    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    #BACK_propagation  hidden to output
    def update_H2O(self,wp,t,y,z,lmda):
        for i in range(0,len(wp)):
            for j in range(0,len(t)):
               self.wp[i][j]-=lmda*(-1)*(t[j]-y[j]*(1-y[j]*z[i]))
        return self.wp
 
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #Back propagation Input to hidden
 
 
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

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def eqation_circle(list_x):
    radious=5
    sq_r=radious*radious
    outcome=(list_x[1]*list_x[1])+(list_x[2]*list_x[2])

    if(outcome<=sq_r):
        target=1
    else:
        target=0
    return target


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@        

def train(w,wp,lmda,outputs,inputs,hidden,examples):

    
    ############################################
    for i in range(0,len(examples)):
        

        x=NN.rand_input(ni)
        z,a=NN.calc_hidout(x,w)
  
        zp,y=NN.calc_output(a,wp)
        
        print("the ouput is :",t)

        target=eqation_circle(x)









    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def test():
    pass

        
if __name__==__main__:
    NN=NN()

    ni=input(int("how many neurons in input:",end=""))
    nh=input(int("how many neurons in hidden layer:",end=""))

    w,wp=NN.init(ni,nh)

    train(w,wp)
    test()


    
    
















                

        

    



































print("how many layers you want in the network:")
L=int(input())


print(L)
n=list()
for i in range(0,L):
    
    n[i]=int(input("number of neurons needed in hidden layer:"))






m=int(input("enter X matrix number of rows :"))
n=int(input("enter X matrix number of columns :"))
x=[]
for i in range(0,m):
    for j in range(0,n):
        x[i][j]=input("give values for X"+"["+i+"]"+"["+j+"]")
        
        
class NN():
    def multip():
        for i in range(len(X)):  
            for j in range(len(Y[0])):  
                for k in range(len(Y)):
                    result[i][j] += X[i][k] * Y[k][j]


for r in result:
    print(r) 
        

class NeuralNetwork:
    
     
    def __init__(self, x, y):
        self.input      = 
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
