##训练集和测试集
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve,auc
from scipy import interp
import os


def get_hbp(topo_root,bc0,bc1,bc2,labs,str,view):
    """load topo feature and labels
        args: topo_root: the folder path 

    """
    path_list = os.listdir(topo_root)
    path_list.sort(key=lambda x:int(x[:-10]))####序列命名不同
    for file in path_list:
        ##get topo feature path
        filename = topo_root[:10]+view+topo_root[13:] +'/'+file
       
        if file.endswith('{}0.npy'.format(str)):
            bc0.append(np.load(filename))
            if file[-9:-8] =='1':
                labs.append(1)
            else:
                labs.append(0)
        elif file.endswith('{}1.npy'.format(str)):
            bc1.append(np.load(filename))
        elif file.endswith('{}2.npy'.format(str)):
            bc2.append(np.load(filename))
    
        
    return bc0,bc1,bc2,labs

##通过load 特征文件（xx.npy）得到feature和labels
topo_root = r'/data/pst/PVP-TOPO/new'
bc0_hbp,bc1_hbp,bc2_hbp,labs_hbp = [],[],[],[]
bc0_hbp,bc1_hbp,bc2_hbp,labs_hbp = get_hbp(topo_root,bc0_hbp,bc1_hbp,bc2_hbp,labs_hbp,'bc','HBP')
topo_root = r'/data/pst/PVP-TOPO/old'
bc0_hbp,bc1_hbp,bc2_hbp,labs_hbps = get_hbp(topo_root,bc0_hbp,bc1_hbp,bc2_hbp,labs_hbp,'bc','HBP')

##组合想要分类的特征
data = np.concatenate((bc0_hbp,bc1_hbp,bc2_hbp),axis=1)
label = labs_hbp

##接下来进行分类
###下面是pipeline+gridsearchcv    
from sklearn.preprocessing   import MinMaxScaler
from sklearn.pipeline        import Pipeline
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.datasets import load_iris
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score   # 准确率
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import roc_curve , auc
import xgboost as xgb
import random
from sklearn.metrics import precision_recall_curve
from sklearn.decomposition import PCA
filefolder = 'PVP_HBP_T2W/'##保存的文件夹
KF = KFold(n_splits = 5)##五折
tprs=[]
aucs=[]
mean_fpr=np.linspace(0,1,100)#在0到1间生成100个点
i=0
#data为数据集,利用KF.split划分训练集和测试集
data = np.array(data)
label = np.array(label)
train_acc=[]
plt.figure()
random.seed(1)
i = 0



##PCA
# pca = PCA()   
# data = pca.fit_transform(data)
# print("降维后",data.shape)
# pca = PCA(n_components="mle")   
# data_fit = pca.fit(data)
# data = data_fit.transform(data)
# print("降维后",data.shape)
# print("特征数",data_fit.explained_variance_ratio_.sum())

##累计贡献曲线
'''
pca.explained_variance_#c查看降维后每个新特征向量所带信息量大小（可解释性方差）
pca.explained_variance_ratio_ #查看降维后每个新特征向量所占信息量占原始信息百分比
pca.explained_variance_ratio_ .sum()# 查看降维后每个新特征向量所占信息量占原始信息百分比的总和
'''
# print(data.shape)
# x = np.linspace(1, 6000, 50)
# pca_line = PCA().fit(data)
# import pdb;
# pdb.set_trace()
# plt.plot(x,np.cumsum(pca_line.explained_variance_ratio_))#np.cumsum分别取前i项的和
# plt.xticks(x) 
# plt.xlabel("number of components after dimension reduction")
# plt.ylabel("cumulative explained variance ratio")
# plt.savefig('1.png')

for train_index,test_index in KF.split(data):##对data进行五折交叉验证
    i = i+1
    #建立模型，并对训练集进行测试，求出预测得分
    #划分训练集和测试集
    X_train,X_test = data[train_index],data[test_index]
    Y_train,Y_test = label[train_index],label[test_index]
    model = SVC(probability=True)
    model.fit(X_train, Y_train)
    #train_acc.append(model.score(X_train, Y_train))
    #print(str(i)+"train_acc"+str(model.score(X_train, Y_train)))
    ##获取训练集的预测值
    ##训练集
    y_pred_train = model.predict(X_train)
    y_pred_proba_train = model.predict_proba(X_train)
   
    fpr_train, tpr_train , threshold_train = metrics.roc_curve(Y_train,y_pred_proba_train[:,1],pos_label = 1)
    ##什么是fpr,tpr,threshold
    ##fpr是被判断未真的错误样本   tpr是判断为真正确的样本
    roc_auc = metrics.auc(fpr_train,tpr_train)
    print("性能评价")
    print(metrics.classification_report(Y_train,y_pred_train))
    maxindex = (tpr_train-fpr_train).tolist().index(max(tpr_train-fpr_train))##找到离对角线最远的点
    threshold = threshold_train[maxindex]##找到对应的threshold
    #print("阈值和对应的值",)
    print("方法1中最好的阈值：",threshold)

    
    lw = 2
    plt.plot(
        fpr_train,
        tpr_train,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.savefig('{}/auc_roc{}.png'.format(filefolder,i))
    plt.clf()


    #利用model.predict获取测试集的预测值
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    print("test----------")
    print(metrics.classification_report(Y_test,y_pred))
    #print("混淆矩阵\n",metrics.confusion_matrix(Y_test,y_pred))
    #计算fpr(假阳性率),tpr(真阳性率),thresholds(阈值)[绘制ROC曲线要用到这几个值]
    fpr, tpr , threshold = metrics.roc_curve(Y_test,y_pred_proba[:,1],pos_label = 1)
    #interp:插值 把结果添加到tprs列表中 
    tprs.append(interp(mean_fpr,fpr,tpr))##更具fpr和tpr预测出100个点的值
    tprs[-1][0]=0.0##设置第一个的tpr为0 
    #计算auc
    roc_auc=auc(fpr,tpr)
    aucs.append(roc_auc)
    #画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
    #plt.plot(fpr,tpr,lw=1,alpha=0.3,label='ROC fold %d(area=%0.2f)'% (i,roc_auc))
 
    i +=1
    print("-------")
##把所有test结果打印出来
plt.figure()
for i in range(len(tprs)):
    plt.plot(mean_fpr,tprs[i],lw=1,alpha=0.3,label='ROC fold %d(area=%0.2f)'% (i,aucs[i]))

#画对角线
plt.plot([0,1],[0,1],linestyle='--',lw=2,color='r',label='Luck',alpha=.8)
mean_tpr=np.mean(tprs,axis=0)
mean_tpr[-1]=1.0
mean_auc=auc(mean_fpr,mean_tpr)#计算平均AUC值
std_auc=np.std(tprs,axis=0)
plt.plot(mean_fpr,mean_tpr,color='b',label=r'Mean ROC (area=%0.2f)'%mean_auc,lw=2,alpha=.8)
std_tpr=np.std(tprs,axis=0)
tprs_upper=np.minimum(mean_tpr+std_tpr,1)
tprs_lower=np.maximum(mean_tpr-std_tpr,0)
plt.fill_between(mean_tpr,tprs_lower,tprs_upper,color='gray',alpha=.2)
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc='lower right')
plt.savefig('{}/test_auc_roc.png'.format(filefolder))
plt.clf()    
