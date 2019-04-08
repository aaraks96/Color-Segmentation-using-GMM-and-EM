import numpy as np
import cv2
import copy
import sys
import matplotlib.pyplot as plt
import math
import os
from imutils import contours


def getData(folder_name):
    data = []
    for iter1 in os.listdir(folder_name):           #
        img = cv2.imread(os.path.join(folder_name,iter1))
        resized = cv2.resize(img,(40,40),interpolation=cv2.INTER_LINEAR)
        img = resized[13:27,13:27]

        nx = img.shape[0]
        ny = img.shape[1]
        ch = img.shape[2]

        img = np.reshape(img,(nx*ny,ch))
        for i in range(img.shape[0]):
            data.append(img[i,:])

    return np.array(data)

        
def GMM(data,K):
    
    n_feat = data.shape[0] 
    n_obs = data.shape[1] 
    
   
    
    def gaussian(x,mean,cov):
        det_cov = np.linalg.det(cov)
        cov_inv = np.zeros_like(cov)
        for i in range(n_obs):
            cov_inv[i,i] = 1/cov[i,i]
        diff = np.matrix(x-mean)
        
        N = (2.0 * np.pi) ** (-len(data[1]) / 2.0) * (1.0 / (np.linalg.det(cov) ** 0.5)) *\
            np.exp(-0.5 * np.sum(np.multiply(diff*cov_inv,diff),axis=1))
        return N
    
    
    def initialize():
        mean = np.array([data[np.random.choice(n_feat,1)]],np.float64)
        cov = [np.random.randint(1,255)*np.eye(n_obs)]
        cov = np.matrix(np.multiply(cov,np.random.rand(n_obs,n_obs)))
        return {'mean': mean, 'cov': cov}
   
   
   
    bound = 0.0001
    max_itr = 500
    
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
        
        cluster_prob = np.divide(cluster_prob,np.tile(cluster_sum,(K,1)).transpose())
        
        Nk = np.sum(cluster_prob,axis = 0) #2
        
        
        #EM - step M
        for cluster in range (K):
            temp_sum = math.fsum(cluster_prob[:,cluster])
            new_mean = 1./ Nk[cluster]* np.sum(cluster_prob[:,cluster]*data.T,axis=1).T
            
            parameters[cluster]['mean'] = new_mean
            diff = data - parameters[cluster]['mean']
            new_cov = np.array(1./ Nk[cluster]*np.dot(np.multiply(diff.T,cluster_prob[:,cluster]),diff)) 
            parameters[cluster]['cov'] = new_cov
            mix_c[cluster] = 1./ n_feat * Nk[cluster]
            
            
       
        if len(log_likelihoods)<2: continue
        if np.abs(log_likelihood-log_likelihoods[-2])<bound : break
    
    return mix_c,parameters


green_train_data = getData(folder_name="green_train")
orange_train_data = getData(folder_name="orange_train")
yellow_train_data = getData(folder_name="yellow_train")

mix_c,parameters = GMM(green_train_data, 4)
np.save('weights_g.npy',mix_c)
np.save('parameters_g.npy',parameters)
mix_c,parameters = GMM(orange_train_data, 6)
np.save('weights_o.npy',mix_c)
np.save('parameters_o.npy',parameters)
mix_c,parameters = GMM(yellow_train_data, 7)
np.save('weights_y.npy',mix_c)
np.save('parameters_y.npy',parameters)

def test(frame,K,weights,parameters,div,color,r):
    
    
    
    def gaussian(data,mean,cov):
        det_cov = np.linalg.det(cov)
        cov_inv = np.linalg.inv(cov)
        diff = np.matrix(data-mean)
        
        
        N = (2.0 * np.pi) ** (-len(data[1]) / 2.0) * (1.0 / (np.linalg.det(cov) ** 0.5)) *\
            np.exp(-0.5 * np.sum(np.multiply(diff*cov_inv,diff),axis=1))
        
        return N
    
    test_image = frame
    nx = test_image.shape[0]
    ny = test_image.shape[1]
    img = test_image
    ch = img.shape[2]
    img = np.reshape(img, (nx*ny,ch))
    
    #weights = np.load('weights_o.npy')
    #parameters = np.load('parameters_o.npy')
    prob = np.zeros((nx*ny,K))
    likelihood = np.zeros((nx*ny,K))
    
    for cluster in range(K):
       prob[:,cluster:cluster+1] = weights[cluster]*gaussian(img,parameters[cluster]['mean'], parameters[cluster]['cov'])
       
       likelihood = prob.sum(1)
       
    
    probabilities = np.reshape(likelihood,(nx,ny))
    
    probabilities[probabilities>np.max(probabilities)/div] = 255
    
    
    
    output = np.zeros_like(frame)
    output[:,:,0] = probabilities
    output[:,:,1] = probabilities
    output[:,:,2] = probabilities
    blur = cv2.GaussianBlur(output,(3,3),5)
    #cv2.imshow("out",output)
    edged = cv2.Canny(blur,50,255 )
    
    cnts,h = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (cnts_sorted, boundingBoxes) = contours.sort_contours(cnts, method="left-to-right")
    
    hull = cv2.convexHull(cnts_sorted[0])
    (x,y),radius = cv2.minEnclosingCircle(hull)
    
    if radius > r:
        cv2.circle(test_image,(int(x),int(y)-1),int(radius+1),color,4)

        #cv2.imshow("Final output",test_image)
        return test_image
    else:
        #cv2.imshow("Final output",test_image)
        return test_image

images = []
video = "detectbuoy.avi"
cap = cv2.VideoCapture(video)
weights_o = np.load('weights_o.npy')
parameters_o = np.load('parameters_o.npy')
weights_g = np.load('weights_g.npy')
parameters_g = np.load('parameters_g.npy')
weights_y = np.load('weights_y.npy')
parameters_y = np.load('parameters_y.npy')

while (cap.isOpened()):
    success, frame = cap.read()
    if success == False:
        break

    test(frame,4,weights_g,parameters_g,div= 8.5, color=(0, 255, 0),r = 9)
    test(frame, 7,weights_y,parameters_y,div= 9.5, color=(0, 255, 255),r= 7)
    test(frame,6,weights_o,parameters_o,div= 3.0,color=(0, 128, 255),r = 7)

    images.append(frame)           
    cv2.imshow("output", frame)         
    cv2.waitKey(5)                      


### Writing to a video
source = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('3D_gauss_all.avi', source, 5.0, (640, 480))
for image in images:
    out.write(image)
    cv2.waitKey(10)

out.release()
cap.release()


