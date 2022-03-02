import numpy as np
import cv2
from queue import Queue
from realsense_depth import *
import matplotlib.pyplot as plt

def Image_cleaning(img):
    img=cv2.GaussianBlur(img, (5, 5), cv2.BORDER_CONSTANT)
    kernel = np.ones((19,19),np.uint8)
    img=cv2.dilate(img,kernel,iterations=1) 

    return img


def Create_Histogram(img,bins,range):
    histogram, bin_edges = np.histogram(img, bins, range)
    return histogram, bin_edges

def plot_histogram(histogram, bin_edges):
    plt.figure()
    plt.title("Histogram")
    plt.xlabel("Depth")
    plt.ylabel("pixel count")
    plt.xlim([0.0, 50000.0])  

    plt.plot(bin_edges[0:-1], histogram)  
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def Process_Histogram(histogram, bin_edges,sigma,threshold,bin_threshold=0):
    
    Sbin_ind=np.argmax(histogram)
    region_density=0
    for i in range(sigma):
        if (Sbin_ind-i >=0):
            region_density+=histogram[Sbin_ind-i]
        if (Sbin_ind+i <=bin_edges.shape[0]-1):
            region_density+=histogram[Sbin_ind+i]
    if (region_density/(np.sum(histogram))>=threshold):
        is_wall=True
    else:
        is_wall=False
    if (Sbin_ind<bin_threshold):
        is_wall=False
    # print(histogram)
    # print(bin_edges)
    bin_averages=np.zeros(bin_edges.shape[0]-1)
    for i in range(bin_averages.shape[0]):
        bin_averages[i]=(bin_edges[i]+bin_edges[i+1])/2
    weighted_mean=np.dot(histogram,bin_averages)/(np.sum(bin_averages))
    
    return is_wall,weighted_mean

def Update_queue(q,size, element):
    if (len(q)<size):
        q.append(element)
        
    else:
        q.pop()
        q.append(element)

def Mean_queue(q,size):
    if (len(q)<size):
        return 0
    else:
        return sum(q)/len(q)
        

if __name__ == "__main__":
    dc = DepthCamera()
    while True:
        
        ##Read frame
        ret, depth_frame, color_frame = dc.get_frame()
        
        
        ##Image Pre-processing
        depth_frame2=Image_cleaning(depth_frame)
        
        
        # ##Histogram Formation
        histogram, bin_edges=Create_Histogram(depth_frame2,50,(0,50000))
        
        # plot_histogram(histogram, bin_edges)
        # print(histogram)
        # print('###')
        # print(np.argmax(histogram))
        # print(np.flip(np.argsort(histogram)))
        # print(np.sum(histogram))
        
        ##Histogram processing
        is_wall,weighted_mean=Process_Histogram(histogram, bin_edges,1,0.8,3)
        print("Wall:",is_wall,"     Mean Depth:",depth_frame2.max())
        ##Show Image
        # cv2.imshow("Color_frame_original", color_frame)
        cv2.imshow("depth frame_original", depth_frame2)
        # cv2.imshow("depth frame_processed", depth_frame2)
        key = cv2.waitKey(1)
        if key == ord('s'):
            break
