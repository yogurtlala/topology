import matplotlib.colors
import numpy as np
import math
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import SimpleITK as sitk
from PIL import Image
import gudhi as gd
import gudhi.representations
import os 
import datetime
from trans import GetRoi,GetRoi_topo
from gudhi.representations import (DiagramSelector, Clamping, Landscape, Silhouette, BettiCurve, ComplexPolynomial,\
  TopologicalVector, DiagramScaler, BirthPersistenceTransform,\
  PersistenceImage, PersistenceWeightedGaussianKernel, Entropy, \
  PersistenceScaleSpaceKernel, SlicedWassersteinDistance,\
  SlicedWassersteinKernel, PersistenceFisherKernel, WassersteinDistance)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

"""as an example,we extract the landscape,betti_curve,entropy from img.
    (you can try other topo representation)
    get_topo():an plug-and-play modules to get topo feature
    main():extract topo feature from file_folder and save in file.
"""
def Diagram(D1):##输入interval展示landscape
    proc1 = DiagramSelector(use=True, point_type="finite")
    D1 = proc1(D1)
    print(D1)
    
    return D1
def landscape2(D1,data_nii):##输入是个数组
    LS = Landscape(resolution=1000)
    L = LS(D1)##输出是个数组

    return L
def betti_curve(D1,data_nii):
    BC = BettiCurve(resolution=1000)
    ##保存文件，检查曲线是否正常
    # plt.plot(BC(D1))
    # plt.title("Betti Curve")
    # plt.savefig('{}.png'.format(str(data_nii[:-4])))
    # plt.clf()
    return BC(D1)
def entropy(D1,data_nii):
    ET = Entropy(mode="scalar")
    #print("Entropy statistic is " + str(ET(D1)))\
    return ET(D1)

def get_img_as_gray_array(image_path):

    image_obj = sitk.ReadImage(image_path)
    image_data = sitk.GetArrayFromImage(image_obj)
    return image_data
def tranform_img_to_bitmap(image, factor=0):
    # #存储3d图像的bitmap
    
    bitmap = []
    for i in range(0, image.shape[0] - factor):  
        for j in range(0, image.shape[1] - factor):
            for k in range(0,image.shape[2]-factor):
                bitmap.append(image[i, j, k])  ###将image保存在bitmap中

    return bitmap
from sklearn.preprocessing import MinMaxScaler

def minmax_normalization(data,max_val = 256):
    
    data1 = data - np.amin(data)
    data2 = np.amax(data)- np.amin(data)
    norm_img = data1 / data2
    norm_img = norm_img * max_val
    norm_img = np.floor(norm_img)
    norm_img = norm_img.astype(int)
    
    return norm_img
   
def get_persistence_diagrams_for_img(gray):##输入图片，得到dgm

    
    gray = minmax_normalization(gray)
    bitmap = tranform_img_to_bitmap(gray)###事实上 这个只是个原图
    bcc = gd.CubicalComplex(top_dimensional_cells=bitmap, dimensions=[gray.shape[0], gray.shape[1],gray.shape[2]])
    bcc.compute_persistence(5)##设置最小persistence

    return bcc.persistence_intervals_in_dimension(0),bcc.persistence_intervals_in_dimension(1),bcc.persistence_intervals_in_dimension(2)

def read_img(img_name):
    """Under an file folder get topo representation one by one
        args:img_name:the img folder path
        return:topo.npy which save the topo representation of [l](l decided by resolution and feature)
    """
    i = 0
    path_list = os.listdir(img_name)
    path_list.sort(key=lambda x:int(x[:-9]))
    for filename in path_list: 
        ##get img path
        filename1 = img_name+'/'+filename
        print(filename1)
        ##get img array
        gray = GetRoi_topo(filename1)
        ##get persistence_intervals_in_dimension
        dgm = get_persistence_diagrams_for_img(gray)
        dgm0 , dgm1 , dgm2 = np.array(dgm)
        dgm0 = Diagram(dgm0)
        dgm1 = Diagram(dgm1)
        dgm2 = Diagram(dgm2)
        ##extract the feature with different dimension and representation
        ld0 = landscape2(dgm0,filename)
        bc0 = betti_curve(dgm0,filename)
        np.save('/data/pst/T2W-TOPO/old/{}_bc0.npy'.format(filename[:-7]),bc0)
        np.save('/data/pst/T2W-TOPO/old/{}_ld0.npy'.format(filename[:-7]),ld0)

def get_topo(img):
    """an plug-and-play modules to get topo representation 
        args:img of [h,w,d]
        return: topo representation of [l](decided by the resolution and feature number)
    """

    dgm =  get_persistence_diagrams_for_img(img)  
    dgm0 , dgm1 , dgm2 = np.array(dgm)
    dgm0 = Diagram(dgm0)
    dgm1 = Diagram(dgm1)
    dgm2 = Diagram(dgm2)

    ##get topo feature with different dimension and representation
    filename = '1111'##the path of topo representation picture
    ld0 = landscape2(dgm0,filename)
    bc0 = betti_curve(dgm0,filename)
    ld1 = landscape2(dgm1,filename)
    bc1 = betti_curve(dgm1,filename)
    ld2 = landscape2(dgm2,filename)
    bc2 = betti_curve(dgm2,filename)
    return np.concatenate((bc0,bc1,bc2),axis = 0)

def main():
    ###img folder path
    content_img_path = "/data/pst/T2W-feature/old"
    read_img(content_img_path)#


