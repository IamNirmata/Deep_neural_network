# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 20:17:33 2019

@author: sreeh
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 01:33:00 2019

@author: Sreenath
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 10:10:43 2019

@author: HARI
"""
import mpmath
from numpy import zeros
import numpy  as np
import random
import math
from tqdm import tqdm

class bpnn(object):

    def init(self,ni,nh):
        ############################################################ 
        #w is randomly defined
        self.w= zeros([nh,ni+1])
        
        for i  in range(0,nh):
            
            for j in range(0,ni+1):
                self.w[i][j]=random.uniform(-1,1)
                

        #############################################################
        #wp is randomly definned
        
        self.wp= zeros([1,nh+1])
        
        for i  in range(0,nh+1):
            
            self.wp[0][i]=random.uniform(-1,1)


        ###########################################################
        return self.w,self.wp


    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def rand_input(self,ni):
        #input x is randomly defined
        self.x= zeros([ni+1,1])
        
        for i in range(0,ni+1):
            if(i==0):
                self.x[i][0]=1         
            else:
                self.x[i][0]=random.randint(-5,5)
        
        
        
        ##############################################################
        return self.x
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




    def calc_hidout(self,x,w):
        self.z= zeros([len(self.w),1])
        
        
        for i in range(0,len(w)):
            for j in range(0,len(w[0])):
                
                self.z[i][0]+=self.w[i][j]*self.x[j][0]
                
        

        #adding bias as a[0]=1
        k=len(self.z)+1
        self.a= zeros([k,1])
        self.a[0][0]=1

        

        for i in range (0,len(self.z)):
            z_int=(self.z[i][0])
            self.a[i+1][0]= (1 / (1 + np.exp(-(z_int))))
            
            #np.exp(np.array([1391.12694245],dtype=np.float128))*100
            
            
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
        
        self.zp=0
        for i in range(0,len(wp[0])):

            self.zp+=self.wp[0][i]*self.a[i]

            
        self.y=  1 / (1 + math.exp(-self.zp))
        

        return self.zp,self.y







    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    #BACK_propagation  hidden to output
    def update_H2O(self,wp,t,y,a,lmda):
        for i in range(0,len(wp[0])):
               self.wp[0][i]-=lmda*(-1)*(t-y)*y*(1-y)*a[i][0]
        return self.wp
 
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #Back propagation Input to hidden
 
 
    def update_I2H(self,output_errors,wp,w,x,z,lmda):
        self.hidden_errors= zeros([len(wp[0]),1])
        
        for i in range(0,len(wp[0])):
            self.hidden_errors[i][0]=0
            
            self.hidden_errors[i][0]+=wp[0][i]*output_errors
            
            
        
#**************************************************************************
        for i in range(0,len(w)):
            for j in range(0,len(w[0])):
                self.delta=(-1)*self.hidden_errors[i][0]*self.z[j][0]*(1-z[i][0])*x[j][0]
                self.w[i][j]+=lmda*self.delta
        
        
        return self.delta

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

        


        wp=NN.update_H2O(wp,t,y,a,lmda)
        delta=NN.update_I2H(output_error,wp,w,x,z,lmda)
        
        
    print("training is done!!!!!")
"""
def testing(w,wp,lmda,epoch):

    
    ############################################
    for i in range(0,len(epoch)):
        

        x=NN.rand_input(ni)
        z,a=NN.calc_hidout(x,w)
  
        zp,y=NN.calc_output(a,wp)
        
        print("the ouput is :",y)
        
        if(y>0.8):
            output=1
        else:
            output=0
            
        
        
        print("converted output is",output)
        

        t=eqation_circle(x)

        output_error= t-output

        print("output_error :",output_error)


        wp=NN.update_H2O(wp,t,y,z,lmda)
        w,delta=NN.update_I2H(output_error,wp,w,x,z,lmda)
    print("##############################################")
    print("training is done!!!!!")
    print("")
    print("training is done!!!!!")


"""


    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def test():
    pass

        
if __name__=='__main__':
    NN=bpnn()

    ni=2
    nh=60
    lmda=0.01
    epoch=int(input("No of training iterations:"))


    w,wp=NN.init(ni,nh)

    train(w,wp,lmda,epoch)
    #test()

