#!/usr/bin/env python
# coding: utf-8

# In[27]:


import math
import numpy as np
print("Enter the Gaussian Kernal size")
n=int(input())
x = np.arange(-math.floor(n/2), math.floor((n+1)/2))
y = np.arange(-math.floor(n/2), math.floor((n+1)/2))
a,b= np.meshgrid(x,y);
# print(len(a),len(b)) 


# In[29]:


gaussian_mask=[[0 for i in range(n)]for j in range(n)]
sigma=1.5
for i in range(n):
    for j in range(n):
        p=a[i][j]
        q=b[i][j]
        c=(1/(2*3.14*sigma*sigma))
        d=-((p*p)+(q*q))/(2*sigma*sigma)
        gaussian_mask[i][j]=c*math.exp(d)


# In[31]:


import cv2
import copy
import math
import numpy as np
gray_image=cv2.imread("C:/Users/Devashi Jain/Desktop/IIIT-D/Computer Vision/Assignment-2/CV Assignment 2/Q1.jpeg",0)
#gray_copy=copy.deepcopy(gray_image)
cv2.imshow("Original Image",gray_image)
padding=math.ceil(n/2)-1
def Gaussian_filtering(gray_image,gaussian_mask):
    avg=0
    gray_scale_image=np.zeros((len(gray_image)+(2*padding),len(gray_image[0])+(2*padding)))
    for i in range(len(gray_image)):
        for j in range(len(gray_image[0])):
            gray_scale_image[i+padding][j+padding]=gray_image[i][j]   
    gray_gaussian_image=np.zeros((len(gray_scale_image),len(gray_scale_image[0])))
    for i in range(n):
        for j in range(n):
            avg=avg+gaussian_mask[i][j]
    for i in range(padding,len(gray_scale_image)-padding):
        for j in range(padding,len(gray_scale_image[0])-padding):
            s=0
            for k in range(len(gaussian_mask)):
                for l in range(len(gaussian_mask)):
                    x=abs(k-i-padding)
                    y=abs(l-j-padding)
                    s=s+(gaussian_mask[k][l]*gray_scale_image[x][y])
            s=s/avg
            gray_gaussian_image[i][j]=s
    gray_gaussian_image=gray_gaussian_image[padding:-padding, padding:-padding]
    return gray_gaussian_image
blur_image =Gaussian_filtering(gray_image,gaussian_mask)
blur1=copy.deepcopy(blur_image)
blur1=blur1.astype(np.uint8)
cv2.imshow("implementation of gaussian blur",blur1)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[32]:


sobel_x=[[-1,-2,-1],[0,0,0],[1,2,1]]
sobel_y=[[-1,0,1],[-2,0,2],[-1,0,1]]
X_gradient=copy.deepcopy(blur_image)
Y_gradient=copy.deepcopy(blur_image)
for i in range(1,len(blur_image)-1):
    for j in range(1,len(blur_image[0])-1):
        sx=0
        sy=0
        for k in range(len(sobel_x)):
            for l in range(len(sobel_x)):
                x=abs(k-i-1)
                y=abs(l-j-1)
                sx=sx+(sobel_x[k][l]*blur_image[x][y])
                sy=sy+(sobel_y[k][l]*blur_image[x][y])
        X_gradient[i][j]=sx
        Y_gradient[i][j]=sy
for i in range(len(Y_gradient)):
    for j in range(len(Y_gradient[0])):
        if Y_gradient[i][j]<0:
            Y_gradient[i][j]=Y_gradient[i][j]*(-1)
for i in range(len(Y_gradient)):
    for j in range(len(Y_gradient[0])):
        if Y_gradient[i][j]>255:
            Y_gradient[i][j]=255
for i in range(len(X_gradient)):
    for j in range(len(X_gradient[0])):
        if X_gradient[i][j]<0:
            X_gradient[i][j]=X_gradient[i][j]*(-1)
for i in range(len(X_gradient)):
    for j in range(len(X_gradient[0])):
        if X_gradient[i][j]>255:
            X_gradient[i][j]=255
I_xx=X_gradient*X_gradient
I_yy=Y_gradient*Y_gradient
#print(I_xx)
Edge_Detected=[]
Edge_Detected=I_xx+I_yy
for i in range(len(Edge_Detected)):
    for j in range(len(Edge_Detected[0])):
        Edge_Detected[i][j]=math.sqrt(Edge_Detected[i][j])
        if Edge_Detected[i][j]>255:
            Edge_Detected[i][j]=255
# for i in range(len(I_xx)):
#     for j in range(len(I_xx[0])):
#         Edge_Detected[i][j]=math.sqrt((I_xx[i][j]+I_yy[i][j]))
edge=copy.deepcopy(Edge_Detected)
edge=edge.astype(np.uint8)
cv2.imshow("Edge Detected",edge)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[61]:


X_coordinates=[]
Y_coordinates=[]
Detected_Edge=copy.deepcopy(Edge_Detected)
for i in range(len(Detected_Edge)):
    for j in range(len(Detected_Edge[0])):
        if(Detected_Edge[i][j]>=60):
            Detected_Edge[i][j]=255
            X_coordinates.append(i)
            Y_coordinates.append(j)
        else :
            Detected_Edge[i][j]=0
Detected=Detected_Edge.astype(np.uint8)
cv2.imshow("Edge Detected",Detected)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[62]:


color_image=cv2.imread("C:/Users/Devashi Jain/Desktop/IIIT-D/Computer Vision/Assignment-2/CV Assignment 2/Q1.jpeg",0)
cv2.imshow("Circle Detected",color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[63]:


import math
rmin=15
rmax=int(np.min(Detected_Edge.shape)/2)
count=[]
for i in range(rmax):
    temp2=[]
    for j in range(Detected_Edge.shape[0]):
        temp=[]
        for k in range(Detected_Edge.shape[1]):
            temp.append(0)
        temp2.append(temp)
    count.append(temp2)
print(len(count))


# In[64]:


try:
    for radius in range(rmin,rmax):
        for x in range(len(X_coordinates)):
            theta=0
            while theta < 360:
                a=X_coordinates[x]-radius*(math.cos((theta*np.pi)/180))
                b=Y_coordinates[x]-radius*(math.sin((theta*np.pi)/180))
                if a>=0 and b>=0 and a<Detected_Edge.shape[0] and b<Detected_Edge.shape[1]:
                    count[radius][int(a)][int(b)]+=1
                theta+=10
except:
    print(a,b,radius)


# In[65]:


copyimg=copy.deepcopy(gray_image)


# In[66]:


y=np.max(count)


# In[67]:


for i in range(rmin,rmax):
    for j in range(len(count[0])):
        for k in range(len(count[0][0])):
            if count[i][j][k]>=y*0.7:
                cv2.circle(color_image,(k,j),i,color=(0,0,255))
                


# In[ ]:


#DetectedCircle=copyimg.astype(np.uint8)
cv2.imshow("Circle Detected",color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




