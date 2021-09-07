import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import xlsxwriter as xl
from xlsxwriter import workbook
from xlsxwriter import worksheet
from xlsxwriter.workbook import Workbook
import os.path
import glob

# path= 2
no_pics_compled=0
# s="HUL-57"
# main_path='C:/Project files/Pics/45 days field picture 1st sowing/'+s
# name_for_excel_path='C:/Project files/Pics/45 days field picture 1st sowing'

s="CheckImages"
main_path='C:/Project files/'+s
name_for_excel_path='C:/Project files'
title =['Normalized Red','Normalized green','Normalized blue','Excess red','Excess blue', 'Excess green red', 'Green blue difference','Red blue difference','Red green difference','Green red ratio','Green blue ratio','Normalized green red difference','Normalized green blue difference','Modified Normalized green red difference','Visible band difference','Red green blue vegitation index','Crust index','Color index of vegitation index','Triangular greenness index','Modifed excess green']

MeanTotal =np.zeros(len(title))
sigmaTotal =np.zeros(len(title))
thetaTotal =np.zeros(len(title))
deltaTotal =np.zeros(len(title))

wb=xl.Workbook(s+".csv")
ws=wb.add_worksheet("New")

r=0
c=1
for i in title:
    r=0
    ws.write(r,c,i)
    r=r+1
    ws.write(r,c,"Mean Values")
    c=c+1
    ws.write(r,c,"Sigma Values")
    c=c+1
    ws.write(r,c,"Theta Values")
    c=c+1
    ws.write(r,c,"Delta Values")
    c=c+1  
r=2    
images_list = os.listdir(main_path)
a=[1]
for q in a:
    for p in images_list:
        path=main_path+'/'+p
        print(path)
        img_bgr = cv.imread(path)
        img_bgr_o = cv.imread(path)
        img_color = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

        def Contrast(img):
            "Colour Contrast"
            B = img[..., 0].astype('float32')
            G = img[..., 1].astype('float32')
            R = img[..., 2].astype('float32')

            min_b=np.amin(B)
            min_g = np.amin(G)
            min_r = np.amin(R)

            max_b =np.amax(B)
            max_g = np.amax(G)
            max_r = np.amax(R)
            x,y,z =img.shape
            for i in range(x):
                for j in range(y):
                    bv= ((img[i][j][0]-min_b)*(255))/(max_b-min_b)
                    if min_b>img[i][j][0]:
                        bv=0
                    elif max_b<img[i][j][0]:
                        bv=255

                    gv= ((img[i][j][1]-min_g)*(255))/(max_g-min_g)
                    if min_g>img[i][j][1]:
                        gv=0
                    elif max_g<img[i][j][1]:
                        gv=255

                    rv= ((img[i][j][2]-min_r)*(255))/(max_r-min_r)
                    if min_r>img[i][j][2]:
                        rv=0
                    elif max_r<img[i][j][2]:
                        rv=255


                    img[i][j]=(bv,gv,rv)
            return img

        img_bgr =Contrast(img_bgr)



        def BGR2ExG(img):
            "Excess Green Index"
            b = img[...,0].astype('float32')
            g = img[...,1].astype('float32')
            r = img[...,2].astype('float32')
            ExG = ((2*g) - r - b)  /(r+g+b)
            return ExG

        img_ExG = BGR2ExG(img_bgr)


        def BGR2MExG(img):
            "Modified Excess Green Index"
            B = img[...,0].astype('float32')
            G = img[...,1].astype('float32')
            R = img[...,2].astype('float32')
            MExG = 1.262 * G - 0.884 * R - 0.311 * B
            return MExG
        img_MExG = BGR2MExG(img_bgr)



        def BGR2CIVE(img):
            "Colour Index of Vegetation Extraction"
            B = img[...,0].astype('float32')
            G = img[...,1].astype('float32')
            R = img[...,2].astype('float32')
            CIVE = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745
            return CIVE
        img_CIVE = BGR2CIVE(img_bgr)


        def MExGCIVE(img):
            osimg = BGR2MExG(img) - BGR2CIVE(img)
            return osimg

        img_MExGCIVE = MExGCIVE(img_bgr)


        VIimages =[img_bgr,img_ExG,img_MExG,img_CIVE,img_MExGCIVE]
        VItitles=['img_bgr','img_ExG','img_MExG','img_CIVE','img_MExGCIVE']


        # for i in range(len(VIimages)):
        #     plt.figure(VItitles[i])
        #     plt.imshow(VIimages[i],cmap='gray')
        # plt.show()

        ret, threshExG = cv.threshold(img_ExG,0,255, cv.THRESH_BINARY)
        ret, threshCIVE = cv.threshold(img_CIVE,0,255, cv.THRESH_BINARY_INV)
        ret, threshMExG = cv.threshold(img_MExG,0,255, cv.THRESH_BINARY)
        ret, threshMExGCIVE = cv.threshold(img_MExGCIVE,0,255, cv.THRESH_BINARY)

        threshImages =[threshExG,threshCIVE,threshMExG,threshMExGCIVE]
        threshtiiles =['threshExG','threshCIVE','threshMExG','threshMExGCIVE']

        # for i in range(len(threshImages)):
        #     plt.figure(threshtiiles[i])
        #     plt.imshow(threshImages[i])
        # plt.show()

        img_for_crop = img_bgr

        def croped(im1,im2,im3,im4):
            x,y,z = im1.shape
            for i in range (x):
                for j in range (y):
                    if im2[i,j]<=0 or im3[i,j]<=0 or im4[i][j]<=0 :
                        im1[i,j]=(0,0,0)
                    else:
                        im1[i,j]=im1[i,j]   
            return im1

        img_for_crop = croped(img_bgr,threshMExGCIVE,threshCIVE,threshMExG)

        # plt.figure("cropped")
        # plt.imshow(img_for_crop)
        # plt.figure("non croped")
        # plt.imshow(img_bgr_o)
        # plt.show()


        #Normalization of Image

        norm_img = np.zeros((0,800))
        final_img = cv.normalize(img_for_crop,  norm_img, 0, 255, cv.NORM_MINMAX)

        titles = ['Original', 'cropped image', 'final image']
        images = [img_color, img_for_crop, final_img]

        # for i in range(len(images)):
        #     plt.figure(titles[i])
        #     plt.imshow(images[i])
        # plt.show()


        def VIvalues(im1):
            NR =[]
            NG=[]
            NB=[]
            ExR=[]
            ExB=[]
            ExGR =[]
            GBD =[]
            RBD = []
            RGD=[]
            GRR=[]
            GBR=[]
            NGRD = []
            NGBD = []
            MNGRD =[]
            VD =[]
            RGBVI = []
            CI = []
            CIVE= []
            TGI =[]
            MExG =[]

            x,y,z=im1.shape
            n=0
            for i in range(x):
                for j in range(y):
                    b=im1[i][j][0]
                    g=im1[i][j][1]
                    r=im1[i][j][2]
                    t=b+g+r
                    if(t!=0):
                            NR.append((r / t))
                            NG.append((g/t))
                            NB.append((b/t))
                            ExR.append((((1.4*r)-g)/t))
                            ExB.append((((1.4*b)-g)/t))
                            ExGR.append(((3*g-2.4*r-b) /t))
                            GBD.append((g-b))
                            RBD.append((r-g))
                            RGD.append((r-g))
                            if(r!=0):
                                    GRR.append((g/r))
                            if(b!=0):
                                    GBR.append((g/b))
                            if(g+r)!=0:
                                    NGRD.append(((g-r)/(g+r)))
                            if(g+b)!=0:
                                NGBD.append(((g-b)/(g+b)))
                            if((g*g)+(r*r))!=0:
                                MNGRD.append((((g*g)-(r*r))/((g*g)+(r*r))))
                            if(2*g+b+r) !=0:
                                VD .append(((2*g-b-r)/(2*g+b+r)))
                            if((g*g)+(b*r)) !=0:
                                RGBVI.append((((g*g)-(b*r))/ ((g*g)+(b*r))))
                            if (r+b) !=0:
                                CI.append(((2*b)/(r+b)))
                            CIVE.append((0.441 * r - 0.811 * g + 0.385 * b + 18.78745))
                            TGI.append((95*g - 35*r -60*b))
                            MExG.append((1.262*g - 0.884*r -0.311*b))  

            return [NR,NG,NB,ExR,ExB,ExGR,GBD,RBD,RGD,GRR,GBR,NGRD,NGBD,MNGRD,VD,RGBVI,CI,CIVE,TGI,MExG]
        # NR =[1,2,3,4,5,6,7,8,9,10,11,21,3,6,5,48,585,8,5855,998,74,52,0,14,5,6,235,2,5]
        

        viIndices =VIvalues(final_img)
       
            

        def mean(array):
            mean=[]
            for i in array:
                sum=0
                for j in i:
                    sum+=j
                mean.append(sum/len(i))
            return mean

        mean_values =mean(viIndices) #mean values of all VI's of an image

        def sigma(array,mean):
            sigma=[]
            k=0
            for i in array:
                sum=0
                for j in i:
                    sum = sum+((j-mean[k])**2)
                avg=sum/len(i)
                sigma.append(pow(avg,0.5))
                k=k+1
            return sigma

        sigma_values = sigma(viIndices,mean_values)

        def theta(array,mean,sigma):
            theta=[]
            k=0
            for i in array:
                sum=0
                for j in i:
                    sum=sum+((j-mean[k])**3)
                denominator=len(i) * ((sigma[k])**3)

                final=sum/denominator
                theta.append(final)
                k=k+1
            
            return theta

        theta_values = theta(viIndices,mean_values,sigma_values)

        def delta(array,mean,sigma):
            delta=[]
            k=0
            for i in array:
                sum=0
                for j in i:
                    sum=sum+((j-mean[k])**4)
                denominator=len(i) * ((sigma[k])**4)

                final=sum/denominator
                delta.append(final)
                k=k+1
            
            return delta   

        delta_values=delta(viIndices,mean_values,sigma_values)

        MeanTotal = MeanTotal+mean_values
        sigmaTotal =sigmaTotal+sigma_values
        thetaTotal =thetaTotal+theta_values
        deltaTotal =deltaTotal+delta_values

        c=0
        ws.write(r,c,path.removeprefix(name_for_excel_path+'/'))  
        c=c+1
        for idx in mean_values:
            ws.write(r,c,idx)
            c=c+4
        c=2
        for idx in sigma_values:
            ws.write(r,c,idx)
            c=c+4
        c=3
        for idx in theta_values:
            ws.write(r,c,idx)
            c=c+4
        c=4
        for idx in delta_values:
            ws.write(r,c,idx)
            c=c+4
        c=0
        r=r+1
        no_pics_compled+=1
        print(no_pics_compled)


MeanTotal = MeanTotal/len(images_list)
#[nr,ng,nb..............]

sigmaTotal =sigmaTotal/len(images_list)
thetaTotal =thetaTotal/len(images_list)
deltaTotal =deltaTotal/len(images_list)


wa=wb.add_worksheet("Avg")
r=0
c=1
for i in title:
    r=0
    wa.write(r,c,i)
    r=r+1
    wa.write(r,c,"Mean Values")
    c=c+1
    wa.write(r,c,"Sigma Values")
    c=c+1
    wa.write(r,c,"Theta Values")
    c=c+1
    wa.write(r,c,"Delta Values")
    c=c+1 
r=2
c=1
for i in range(len(MeanTotal)):
    wa.write(r,c,MeanTotal[i])
    c=c+1
    wa.write(r,c,sigmaTotal[i])
    c=c+1
    wa.write(r,c,thetaTotal[i])
    c=c+1
    wa.write(r,c,deltaTotal[i])
    c=c+1
wb.close()


print("d")
