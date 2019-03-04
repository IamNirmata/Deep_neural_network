

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 10:10:43 2019
@author: HARI (sreehari sreenath)
@mentor:James Fuller
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
                w[i][j]=random.uniform(-5,5)
                #w is defined as random numbers in between -10,10

        #############################################################
        #wp is randomly definned
        
        wp= zeros([2,nh+1])
        
        for i  in range(0,nh+1):
            for j  in range(0,2):
                wp[j][i]=random.uniform(-5,5)
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
            elif(i==1):
                x[i][0]=random.randint(-50,50)
            else:
                x[i][0]=random.randint(-50,50)
        
        
        
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
        zp=zeros([2,1])
        for i in range(0,len(wp)):
            for j in range(0,len(wp[0])):
            
            
                #calculating zp
                zp[i][0]+=wp[i][j]*a[i][0]

        #applying sigmoid function to get values in the range of (0,1) 
        y=zeros([2,1])
        y[0][0]=  1 / (1 + np.exp(-zp[0][0]))
        y[1][0]=  1 / (1 + np.exp(-zp[1][0]))

        
        #returning Zprime and output "y"
        return zp,y







    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #BACK_propagation  hidden to output
    #updating the weights
    def update_H2O(self,wp,t1,t2,y,a,lmda):
        for i in range(0,len(wp)):
            # Wprime= lambda * derivative of (t-y)  * activation function of hidden layer 
            wp[0][i]-=lmda*(-1)*(t1-y[0][0])*y[0][0]*(1-y[0][0])*a[i][0]
            wp[1][i]-=lmda*(-1)*(t2-y[1][0])*y[1][0]*(1-y[1][0])*a[i][0]
            return wp
 
    
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #Back propagation Input to hidden
 
    #updating weights for Input to hidden layer
    def update_I2H(self,t1,t2,y,wp,w,x,a,lmda):
        #hidden error is defined as zero matrtix
        hidden_errors= zeros([len(wp[0]),1])
        
        for i in range(0,len(wp[0])):
            
            #calculating the hidden error = Wprime * output error
            hidden_errors[i][0]+=(wp[0][i]*(t1-y[0][0])*y[0][0]*(1-y[0][0]))
            +(wp[1][i]*(t2-y[1][0])*y[1][0]*(1-y[1][0]))

              
#**************************************************************************
        for i in range(0,len(w)):
            for j in range(0,len(w[0])):
                
                #delta= hidden error * derivative of (t-y)* input values with bias
                delta=(-1)*hidden_errors[i][0]*a[j][0]*(1-a[j][0])*x[j][0]
                
                
                
                
                #updating W
                w[i][j]+=lmda*delta
        
        #returning W
        return w

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




def eqation_quadrant(x):

    #line 1
    A=-1
    B=1
    C=50
    
    
    outcome=(A*x[1])+(B*x[2])+C

    if(outcome>0):
        #setting targets
        target1=1
    else:
        target1=0

    #line 2
    A=1
    B=1
    C=-50
    
    
    outcome=(A*x[1][0])+(B*x[2][0])+C

    if(outcome>0):
        #setting targets
        target2=1
    else:
        target2=0

    
    return target1,target2


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@     

# Training of the model
def train(w,wp,lmda,epoch):
    #initializing variables to find the accuracy and other details of model
    count=0
    perc=0
    inarow=0
    m_row=0
        
    ############################################
    #iteration starts
    for i in tqdm(range(0,epoch)):
        
        #calling function to feed random values as input 
        x=NN.rand_input(ni)
        
        #function to calculate hidden layer values
        z,a=NN.calc_hidout(x,w)
        
        # function to find ouput layer values
        zp,y=NN.calc_output(a,wp)
        
        #print("\n",y)
        #################################
        if(y[0][0]>=0.80):
            output1=1
            
        if(y[0][0]<0.2):
            output1=0
        else:
            output1=y[0][0]


        
        if(y[1][0]>=0.80):
            output2=1
            
        if(y[1][0]<0.2):
            output2=0
        else:
            output2=y[1][0]
        ####################################

        #getting the target value to compare the output with actual output
        t1,t2=eqation_quadrant(x)


        output_error=zeros([2,1])
    
        

        output_error[0][0]=t1-output1
        output_error[1][0]=t2-output2


        ######################
        if(output_error[0][0]<0.1 and output_error[1][0]<0.1):
            count+=1
            inarow+=1
            if(inarow>m_row):
                #finding the most in a row function 
                m_row=inarow
        else:
            inarow=0
        ######################
    
        
                
        wp_old=wp   
        #upadating the HIDDEN to OUTPUT layer and fixing weight Wprime
        wp=NN.update_H2O(wp,t1,t2,y,a,lmda)
        #print("updated wp",wp)
        
        
        #Updating INPUT to HIDDEN layer by fixing weight W
        w=NN.update_I2H(t1,t2,y,wp_old,w,x,a,lmda)
        
        
    print("training is done!!!!!")
    
    #calculating the ACCURACY percentage
    perc=(count/epoch)*100
    
    #returning required values
    return perc,count,inarow,m_row,w,wp

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#testig begins
def testing(w,wp,epoch):
    #initializing variables to find the accuracy and other details of model    
    count=0
    perc=0
    inarow=0
    m_row=0
    
    
    ############################################
    #testing iteration begins
    for i in tqdm(range(0,epoch)):
        
        #feeding input values 
        x=NN.rand_input(ni)
        
        #calculaing the hidden layer attributes
        z,a=NN.calc_hidout(x,w)
  
    
        #findiing out output layer neurons values 
        zp,y=NN.calc_output(a,wp)
        
        
        #################################
        if(y[0][0]>=0.80):
            output1=1
            
        if(y[0][0]<0.2):
            output1=0
        else:
            output1=y[0][0]


        
        if(y[1][0]>=0.80):
            output2=1
            
        if(y[1][0]<0.2):
            output2=0
        else:
            output2=y[1][0]
        ####################################

        #getting the target value to compare the output with actual output
        t1,t2=eqation_quadrant(x)


        output_error=zeros([2,1])
    
        

        output_error[0][0]=t1-output1
        output_error[1][0]=t2-output2


        ######################
        if(output_error[0][0]<0.1 and output_error[1][0]<0.1):
            count+=1
            inarow+=1
            if(inarow>m_row):
                #finding the most in a row function 
                m_row=inarow
        else:
            inarow=0
        ######################

        
    
        
    print("testing is done")
    
    #calculating the percentage of ACCURACY
    perc=(count/epoch)*100
    
    return perc,count,inarow,m_row
 


#main()       
if __name__=='__main__':
    #Introdunced NN as a bpnn class
    NN=bpnn()
    #number of input neurons
    ni=2
    
    #number of hidden layer neurons
    nh=4
    
    #LEARNING RATE 
    lmda=.01
    
    #initializing weights
    w,wp=NN.init(ni,nh)
    
    q=1
    
    while(q>0):
        
    
        #number of iterations for training
        epoch=int(input("No of training iterations:"))
        
        q=epoch
        
        #calling the training function
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
        print("+++++++++++++++++TEST_RESULTS++++++++++++++++++++++++++++++++")
        print("Number of examples:",epochT,end=" ")
        print("successful predictions:",countT)
        print("in a row:",inarowT,end=" ")
        print("most in a row:",m_rowT)
        print("                               TEST MODEL ACCURACY:",percT)
        
