"""
作者：ZYJ
日期：2022年04月13日
"""

import cv2
import os

def verProject(image):
    
    img = image.copy()
    (h,w)=img.shape
    total = 0
    verlist = [0 for z in range(0, w)]
    
    for row in range(0,w): 
      for col in range(0,h):
        if(img[col,row]==0): 
          verlist[row]+=1 		
          img[col,row]=255
          total += 1
          
    for row in range(0,w): 
      for col in range((h-verlist[row]),h): 
        img[col,row]=0 
        
    mean = int(total/w)
    
    return mean, verlist
    
    
def horProject(image):
    
    img = image.copy()
    (h,w)=img.shape
    total= 0
    horlist = [0 for z in range(0, h)] 

    for col in range(0,h): 
      for row in range(0,w):
        if img[col,row]==0: 
          horlist[col]+=1 		
          img[col,row]=255
          total += 1

    for col in range(0,h): 
      for row in range(0,horlist[col]): 
        img[col,row]= 0
    cv2.imwrite('E:/XUEXI/YANJIUSHENG/trt/result/1.jpg', img)
    mean = int(total/h)
    return mean, horlist


def getBinary(image_path):
    img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    binary = cv2.adaptiveThreshold(img,255,
                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,25,15)
    rect = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
    rect = cv2.morphologyEx(rect, cv2.MORPH_CLOSE, (2,2))
    mask = cv2.dilate(binary,rect)
    
    mask_not = cv2.bitwise_not(mask)
    binary = cv2.bitwise_and(img,mask)
    
    binary_result = cv2.add(binary,mask_not)
    cv2.imwrite('E:/XUEXI/YANJIUSHENG/trt/result/2.jpg', binary_result)
    return binary_result
    
def roi(verlist, horlist, vermean, hormean):
    
    w_start = 0
    w_end = 0
    
    for row in range(len(verlist)):
        if(verlist[row] >= vermean):
            if(w_start == 0):
                w_start = row
            w_end = row

            
    h_start = 0
    h_end = 0
    
    for col in range(len(horlist)):
        if(horlist[col] >= hormean):
            if(h_start == 0):
                h_start = col
            h_end = col
            
    return w_start, w_end, h_start, h_end
    

def removeContours(image):
    
    img = image.copy()
    (h,w) = img.shape
    
    contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours_list = list(contours)
    
    for i in range(len(contours)):
        if(hierarchy[0, i, 3] < 0):
            cv2.drawContours(img, contours[i], -1, (0, 0, 0), 20)
            
    return img
    
def rectAngle(image):

    img = image.copy()
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        
         
        if(x != 0 and y != 0 and h != img.shape[0] and w != img.shape[1]):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imwrite('E:/XUEXI/YANJIUSHENG/trt/result/4.jpg', img)

    

    

    
