# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 22:35:04 2021

@author: Samed
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 20:29:25 2021

@author: Samed
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


pricerange=pd.read_excel("mobilephoneprice.xlsx")
pricerange_np=np.array(pricerange) #np formuna dönüştürüldü
pricerange_np=np.delete(pricerange_np, [1,3,5,17,18], 1)
pricerange_np=pricerange_np[2:2001:3,:]
pricerange_np_pca= pricerange_np[:,0:14] # tüm satırlar 15.sutuna kadar alındı
pricerange_np_sinif= pricerange_np[:,14] #knn alg için sadece sınıf değerleri alında daha sonra kullnlcak
 

#☻değerlerimizi ölçeklendirdik
scaler=StandardScaler()
scaler.fit(pricerange_np_pca)
scaled_features=scaler.transform(pricerange_np_pca)
scaled_features=pd.DataFrame(scaled_features,columns=('battery_power',
'clock_speed','fc','int_memory','m_dep','mobile_wt','n_cores','pc','px_height','px_width',
'ram','sc_h','sc_w','talk_time'))
scaled_features.head()

#ölçeklendirilmiş özelliklerden  iki tane özelliği aldık(PCA)
PCA=PCA(n_components=(2))
PCA.fit(scaled_features)
Transformed_PCA=PCA.transform(scaled_features)

#sınıf sütununun dahil edilmesi
pca_iki_sinif=np.array(Transformed_PCA) #ölçeklendirilmiş değerler alındı
pca_knn = pd.DataFrame(pca_iki_sinif) 
pca_knn.insert(2,"sinif",pricerange_np_sinif,True) #bu ölçeklendirilmiş degerlere sınıf sütunu dahil edildi

#scatter plot
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('özellik 1', fontsize = 15)
ax.set_ylabel('özellik 2', fontsize = 15)
ax.set_title('2 özellikli PCA', fontsize = 20)
targets = [0,1,2,3]
colors = ['red', 'green', 'blue','yellow']
for target, color in zip(targets,colors):
    indicesToKeep = pca_knn['sinif'] == target
    ax.scatter( pca_knn.loc[indicesToKeep,0]
               , pca_knn.loc[indicesToKeep,1]
               , c = color
               , s = 60)
ax.legend(targets)
ax.grid()
    
    
    
    
    
    
    