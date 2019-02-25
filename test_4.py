
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 10:10:43 2019
@author: HARI
"""

from numpy import zeros
import numpy  as np
import random
import math
from tqdm import tqdm

class bpnn(object):

    def init(self,ni,nh):
        ############################################################ 
        #w is randomly defined
        w= zeros([nh,ni+1])
        
        for i  in range(0,nh):
            
            for j in range(0,ni+1):
                w[i][j]=random.uniform(-50,50)
                

        #############################################################
        #wp is randomly definned
        
        wp= zeros([1,nh+1])
        
        for i  in range(0,nh+1):
            
            wp[0][i]=random.uniform(-50,50)


        ###########################################################
        return w,wp


    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def rand_input(self,ni):
        #input x is randomly defined
        x= zeros([ni+1,1])
        
        for i in range(0,ni+1):
            if(i==0):
                x[i][0]=1         
            else:
                x[i][0]=random.randint(-5,5)
        
        
        
        ##############################################################
        return x
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




    def calc_hidout(self,x,w):
        z= zeros([len(w),1])
        
        
        for i in range(0,len(w)):
            for j in range(0,len(w[0])):
                
                z[i][0]+=w[i][j]*x[j][0]
                
        

        #adding bias as a[0]=1
        k=len(z)+1
        a= zeros([k,1])
        a[0][0]=1

        

        for i in range (0,len(z)):
            z_int=(z[i][0])
            a[i+1][0]= (1 / (1 + np.exp(-(z_int))))
            
            #np.exp(np.array([1391.12694245],dtype=np.float128))*100
            
            
        return z,a

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 


    # Now for our function definition. Sigmoid.
    def sigmoid(self,x):
 
        ans=1 / (1 + math.exp(-x))
        return ans
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    # Sigmoid function derivative.
    def dsigmoid(self,y):
        dans=y * (1 - y)
        return dans
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def calc_output(self,a,wp):
        
        zp=0
        for i in range(0,len(wp[0])):

            zp+=wp[0][i]*a[i][0]

            
        y=  1 / (1 + math.exp(-zp))
        

        return zp,y







    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    #BACK_propagation  hidden to output
    def update_H2O(self,wp,output_error,y,a,lmda):
        for i in range(0,len(wp[0])):
               wp[0][i]-=lmda*(-1)*(output_error)*y*(1-y)*a[i][0]
        return wp
 
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #Back propagation Input to hidden
 
 
    def update_I2H(self,output_errors,wp,w,x,z,lmda):
        hidden_errors= zeros([len(wp[0]),1])
        
        for i in range(0,len(wp[0])):
            
            
            hidden_errors[i][0]+=wp[0][i]*output_errors
            
            
        
#**************************************************************************
        for i in range(0,len(w)):
            for j in range(0,len(w[0])):
                delta=(-1)*hidden_errors[i][0]*z[j][0]*(1-z[i][0])*x[j][0]
                w[i][j]+=lmda*delta
        
        
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
        
        
        
        if(y>0.80):
            output=1
        else:
            output=0

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
        wp=NN.update_H2O(wp,output_error,output,a,lmda)
        #print("updated wp",wp)
        w=NN.update_I2H(output_error,wp,w,x,z,lmda)
        
        
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
        else:
            output=0

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
    nh=30
    lmda=0.01
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
    print("+++++++++++++++++TEST_RESULTS++++++++++++++++++++++++++++++++")
    print("Number of examples:",epochT,end=" ")
    print("successful predictions:",countT)
    print("in a row:",inarowT,end=" ")
    print("most in a row:",m_rowT)
    print("                               TEST MODEL ACCURACY:",percT)
    
