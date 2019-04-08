import numpy as np
import cv2
import copy
import sys
import matplotlib.pyplot as plt
import math
import os

def generateData():
    
    data_1 = np.random.normal(0,2,(50,1))
    data_2 = np.random.normal(3,0.5,(50,1))
    data_3 = np.random.normal(6,3,(50,1))
    # print("data1",data_1)
    # print("data2",data_2)
    # print("data3",data_3)
    data = np.concatenate((data_1,data_2,data_3),axis = 0)
    
    return data

def GMM(data,K):
    
    n_feat = data.shape[0] 
    n_obs = data.shape[1] 
    #n_obs = 1
   
    
    def gaussian(x,mean,cov):
        
        
        cov_inv = 1/cov[0]
        
        diff = x-mean
       
        N = (2.0 * np.pi) ** (-len(data[1]) / 2.0) * (1.0 /cov[0] ** 0.5) *\
            np.exp(-0.5 * np.sum(np.multiply(diff*cov_inv,diff),axis=1))
        N = np.reshape(N,(n_feat,1))
        return N
    

    
    def initialize():
        #mean = np.array([data[np.random.choice(n_feat,1),:]],np.float64)
        mean = np.array([data[np.random.choice(n_feat)]],np.float64)
        range_m = []
        
        cov = [np.random.randint(1,50)*0.1]
        #cov = np.matrix(np.multiply(cov,np.random.rand(n_obs,n_obs)))
        #print(cov)
        return {'mean': mean, 'cov': cov}
   
   
   #generate mean and cov matrix- done
    bound = 0.0001
    max_itr = 1000
    
    parameters = [initialize() for cluster in range (K)]
    cluster_prob = np.ndarray([n_feat,K],np.float64)
    
    #EM - step E
    itr = 0
    mix_c = [1./K]*K
    log_likelihoods = []
    while (itr < max_itr):
        print(itr)
        itr+=1
        for cluster in range (K):
            cluster_prob[:,cluster:cluster+1] = gaussian(data,parameters[cluster]['mean'],parameters[cluster]['cov'])*mix_c[cluster]
            
        
        cluster_sum = np.sum(cluster_prob,axis=1)
        log_likelihood = np.sum(np.log(cluster_sum))
        
        log_likelihoods.append(log_likelihood)
        #print(log_likelihoods)
        cluster_prob = np.divide(cluster_prob,np.tile(cluster_sum,(K,1)).transpose())
        Nk = np.sum(cluster_prob,axis = 0) #2
        #EM - step M
        for cluster in range (K):
            temp_sum = math.fsum(cluster_prob[:,cluster])
            new_mean = 1./ Nk[cluster]* np.sum(cluster_prob[:,cluster]*data.T,axis=1).T
            #print(new_mean.shape)
            parameters[cluster]['mean'] = new_mean
            diff = data - parameters[cluster]['mean']
            new_cov = np.array(1./ Nk[cluster]*np.dot(np.multiply(diff.T,cluster_prob[:,cluster]),diff)) 
            parameters[cluster]['cov'] = new_cov
            mix_c[cluster] = 1./ n_feat * Nk[cluster]
            
            
       #log likelihood
        if len(log_likelihoods)<2: continue
        if np.abs(log_likelihood-log_likelihoods[-2])<bound : break
    
   
   
    
    
    return mix_c,parameters
       
    
data = generateData()
#train_data = getData()
K = 3
mix_c,parameters = GMM(data,K) 
# np.save('weights_1d.npy',mix_c)
# np.save('parameters_1d.npy',parameters)
for i in range(K):
    print("Mean "+str(i+1)+":", parameters[i]['mean'][0])
    print("SD "+str(i+1)+":", parameters[i]['cov'][0])
    
    
    
    
