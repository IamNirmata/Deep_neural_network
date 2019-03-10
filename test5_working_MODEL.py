
# -*- coding: utf-8 -*-


"""
Created on Mon Feb 11 10:10:43 2019
@author: HARI
"""
#importing libraries 
from numpy import zeros
import numpy  as np
import random
import math
from tqdm import tqdm

#making a class for neural network which contains all the functions
class bpnn(object):
    #defining the function to initialize weights W and WP
    def init(self,ni,nh):
        ############################################################ 
        #w is randomly defined
        w= zeros([nh,ni+1])
        
        for i  in range(0,nh):
            
            for j in range(0,ni+1):
                w[i][j]=random.uniform(-10,10)
                #w is defined as random numbers in between -10,10

        #############################################################
        #wp is randomly definned
        
        wp= zeros([1,nh+1])
        
        for i  in range(0,nh+1):
            
            wp[0][i]=random.uniform(-10,10)
            #wp is defined as random numbers in between -10,10

        ###########################################################
        return w,wp
        #returning the weights

        
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    #function for feeding random input values to the model
    def rand_input(self,ni):
        #input x is randomly defined
        x= zeros([ni+1,1])
        
        for i in range(0,ni+1):
            if(i==0):
                x[i][0]=1         
            else:
                x[i][0]=random.randint(-10,10)
        
        
        
        ##############################################################
        return x
        #returning random input values 
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



    #function for calculating hidden output for the hidden layer 
    def calc_hidout(self,x,w):
        z= zeros([len(w),1])
        #initializing Z as zero vector matrix
        
        for i in range(0,len(w)):
            for j in range(0,len(w[0])):
                #calculating Z by multiplying weights by input 
                z[i][0]+=w[i][j]*x[j][0]
                
        

        #adding bias as a[0]=1
        k=len(z)+1
        
        #activation matrix defined as zero vector matrix
        a= zeros([k,1])
        
        # Introducing BIAS for hidden layer and the value given as 1
        a[0][0]=1
        
        

        for i in range (0,len(z)):
            z_int=(z[i][0])
            
            #calculating Activation function by applying sigmoid function 
            a[i+1][0]= (1 / (1 + np.exp(-(z_int))))
            
            #np.exp(np.array([1391.12694245],dtype=np.float128))*100
            
        #returning Z and activation functions    
        return z,a

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 


    # sigmoid function 
    def sigmoid(self,x):
 
        ans=1 / (1 + math.exp(-x))
        return ans
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    # Sigmoid function derivative.
    def dsigmoid(self,y):
        dans=y * (1 - y)
        return dans
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


    #Function for calculating the output layer
    def calc_output(self,a,wp):
        
        #zp is intruced as zero 
        #since only one output, no need for matrix
        zp=0
        for i in range(0,len(wp[0])):
            
            #calculating zp
            zp+=wp[0][i]*a[i][0]

        #applying sigmoid function to get values in the range of (0,1)    
        y=  1 / (1 + np.exp(-zp))
        
        #returning Zprime and output "y"
        return zp,y







    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #BACK_propagation  hidden to output
    #updating the weights
    def update_H2O(self,wp,t,y,a,lmda):
        for i in range(0,len(wp[0])):
               # Wprime= lambda * derivative of (t-y)  * activation function of hidden layer 
               wp[0][i]-=lmda*(-1)*(t-y)*y*(1-y)*a[i][0]
        return wp
 
    
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #Back propagation Input to hidden
 
    #updating weights for Input to hidden layer
    def update_I2H(self,t,y,wp,w,x,a,lmda):
        #hidden error is defined as zero matrtix
        hidden_errors= zeros([len(wp[0]),1])
        
        for i in range(0,len(wp[0])):
            
            #calculating the hidden error = Wprime * output error
            hidden_errors[i][0]+=wp[0][i]*(t-y)
        """           
        print("w iss ",w)   
        print("wp iss ",wp)
        print("z is",z)
        print("x is ",x)
            
        """        
#**************************************************************************
        for i in range(0,len(w)):
            for j in range(0,len(w[0])):
                
                #delta=
                delta=(-1)*hidden_errors[i][0]*a[j][0]*(1-a[j][0])*x[j][0]
                
                #delta=hidden_errors[i][0]* a[j][0] 
                """
                print("delta is   ",delta)
                print("########")
                """
                w[i][j]+=lmda*delta
        
        #print("iteration:+++++++++++++++++++++++")
        return w

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


def train(w,wp,lmda,epoch):
    count=0
    perc=0
    inarow=0
    m_row=0
        
    ############################################
    for i in tqdm(range(0,epoch)):
        

        x=NN.rand_input(ni)
        z,a=NN.calc_hidout(x,w)
  
        zp,y=NN.calc_output(a,wp)
        
        #print("\n",y)
        
        if(y>0.80):
            output=1
            
        if(y<0.2):
            output=0
        else:
            output=y

        t=eqation_circle(x)

        output_error=t-output


        ######################
        if(output_error==0):
            count+=1
            inarow+=1
            if(inarow>m_row):
                m_row=inarow
        else:
            inarow=0
        ######################
    
        
                
            
        #print("wp before update",wp)
        wp=NN.update_H2O(wp,t,y,a,lmda)
        #print("updated wp",wp)
        w=NN.update_I2H(t,y,wp,w,x,a,lmda)
        
        
    print("training is done!!!!!")
    perc=(count/epoch)*100
    
    return perc,count,inarow,m_row,w,wp

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def testing(w,wp,epoch):

    count=0
    perc=0
    inarow=0
    m_row=0
    ############################################
    for i in tqdm(range(0,epoch)):
        

        x=NN.rand_input(ni)
        z,a=NN.calc_hidout(x,w)
  
        zp,y=NN.calc_output(a,wp)
        
        
        
        if(y>0.8):
            output=1
        elif(y<0.2):
            output=0
        else:
            output=y

        t=eqation_circle(x)

        output_error=t-output
        
        
        ######################
        if(output_error==0):
            count+=1
            inarow+=1
            if(inarow>m_row):
                m_row=inarow
        else:
            inarow=0
        ######################

        
    
        
    print("testing is done")
    perc=(count/epoch)*100
    
    return perc,count,inarow,m_row
 


        
if __name__=='__main__':
    NN=bpnn()

    ni=2
    nh=75
    lmda=.05
    epoch=int(input("No of training iterations:"))


    w,wp=NN.init(ni,nh)
    perc,count,inarow,m_row,w,wp=train(w,wp,lmda,epoch)
    print("\n\n\n\n\n\n")
    print("+++++++++++++++++TRAINING_RESULTS++++++++++++++++++++++++++++++++")    
    print("Number of examples:",epoch,end=" ")
    print("successful predictions:",count)
    print("in a row:",inarow,end=" ")
    print("most in a row:",m_row)
    print("                              TRAINING MODEL ACCURACY:",perc)
    
    
    epochT=1000
    percT,countT,inarowT,m_rowT=testing(w,wp,epochT)
    print("++++++++++++++++TEST_RESULTS++++++++++++++++++++++++++++++++")
    print("Number of examples:",epochT,end=" ")
    print("successful predictions:",countT)
    print("in a row:",inarowT,end=" ")
    print("most in a row:",m_rowT)
    print("                               TEST MODEL ACCURACY:",percT)
    
