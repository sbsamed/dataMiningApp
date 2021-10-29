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
pricerange_np=np.delete(pricerange_np, [1,3,5,17,18], 1) #kategorik tipteki kolonların silinmesi
pricerange_np=pricerange_np=pricerange_np[2:2001:3,:]  #2-5-8-11... nolu satırların alınması
satir_sayisi2=pricerange_np.shape[0]
sutun_sayisi2=pricerange_np.shape[1]



#######################standart sapma ve a.ortalamaya göre DEGER DEĞİŞME##########################################
s_sapma=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,])
aritmetikortalama=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,])
median=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,])
def s_sapmabul():
    for x in range(sutun_sayisi2-1):
       s_sapma[x]=np.std(pricerange_np[:,x])

def aritmetikortbul():
    for x in range(sutun_sayisi2-1):
       aritmetikortalama[x]=np.mean(pricerange_np[:,x])
  
def medianbul():
    for x in range(sutun_sayisi2-1):
       median[x]=np.median(pricerange_np[:,x])       

s_sapmabul()
aritmetikortbul()
medianbul()

degistirilendegerler = np.empty((0,4), float)
for y in range(sutun_sayisi2-1):
    a1=median[y]+(2*s_sapma[y]) 
    a2=median[y]-(2*s_sapma[y])
    for x in range(satir_sayisi2):
        deger=pricerange_np[x,y]
        if(deger>a1)or(deger<a2):
            pricerange_np[x,y]=aritmetikortalama[y]
            degistirilendegerler = np.append(degistirilendegerler, np.array([[x,y,deger,aritmetikortalama[y]]]), axis=0)

#################################################################

#PCA

pricerange_np_pca= pricerange_np[:,0:14] # 15.sutuna kadar tüm satırlar alındı
pricerange_np_sinif= pricerange_np[:,14] #knn alg için sadece sınıf değerleri alında daha sonra kullnlcak


#değerlerimizi ölçeklendirdik
scaler=StandardScaler()
scaler.fit(pricerange_np_pca)
scaled_features=scaler.transform(pricerange_np_pca)
scaled_features=pd.DataFrame(scaled_features,columns=('battery_power', 
'clock_speed','fc','int_memory','m_dep','mobile_wt','n_cores','pc','px_height','px_width',
'ram','sc_h','sc_w','talk_time')) #ölçeklendirme sonucu oluşan tablodaki sütun adları
scaled_features.head()


#ölçeklendirilmiş özelliklerden  iki tane özelliği aldık(PCA)
PCA=PCA(n_components=(2)) #kaç adet özellik istiyorsak onu belirtiyoruz
PCA.fit(scaled_features) 
Transformed_PCA=PCA.transform(scaled_features)

pca_iki_sinif=np.array(Transformed_PCA)
pca_knn = pd.DataFrame(pca_iki_sinif)
pca_knn.insert(2,"sinif",pricerange_np_sinif,True)
pca_knn=np.array(pca_knn)

#KNN ALGORİTMASI(DERSTE YAZILAN)
satir_sayisi=pca_knn.shape[0]
sutun_sayisi=pca_knn.shape[1]
pca_knn_shuffled=pca_knn

def oklid_uzaklik(v1,v2):
    col_sayi=len(v1) #v1 vektörünün uzunluğu
    t=0
    for i in range(col_sayi):
        t+=(v1[i]-v2[i])*(v1[i]-v2[i])
    
    return np.sqrt(t) #toplamin karaköküne dönüyor    

def ozellik_normallestir(col):
    the_max=np.max(col)
    the_min=np.min(col)
    for i in range(len(col)):
        col[i]=(col[i]-the_min)/(the_max-the_min)#istenirse buraya the_max-the_min'in 0 olma exception'i eklenebilir
    return col

for i in range(2):
    pca_knn_shuffled[:,i]=ozellik_normallestir(pca_knn_shuffled[:,i])

egitim_set=pca_knn_shuffled[:333,:]#ilk 333 satiri egitim setine ayiriyoruz (bu yaklasik %50'lik bir kisim)
egitim_X=egitim_set[:,:sutun_sayisi-1]
egitim_Y=egitim_set[:,sutun_sayisi-1]
egitim_num=egitim_X.shape[0]

val_set=pca_knn_shuffled[333:500,:] #44 satiri validasyon setine ayiriyoruz (bu yaklasik %25'lik bir kisim)
val_X=val_set[:,:sutun_sayisi-1]
val_Y=val_set[:,sutun_sayisi-1]
val_num=val_X.shape[0]

test_set=pca_knn_shuffled[500:,:]  #165 satiri test setine ayiriyoruz (bu yaklasik %25'lik bir kisim)
test_X=test_set[:,:sutun_sayisi-1]
test_Y=test_set[:,sutun_sayisi-1]
test_num=test_X.shape[0]



aday_k=[1,3,5,7,9,11,13,15,17,19,21,23]
performanslar=[] 
for k in aday_k: #aday_k listesinin icini geziyor
    tahminler=[]# her bir validasyon ornegiicin urettigimiz sinif tahminini bu listede tutacagiz.
    
    for v in range(val_num):
        sinifi_merak_edilen=val_X[v,:] #bunu siniflandiracagiz
        uzakliklar=[]#bu liste sinifini merak ettigimiz validasyon orneginin tüm egitim örneklerine olan uzakliklarini tutacak
        for e in range(egitim_num): #her bir egitim ornegi icin
            test_edilen=egitim_X[e,:]
            uzaklik=oklid_uzaklik(sinifi_merak_edilen,test_edilen)
            uzakliklar.append(uzaklik) # e. siradaki egitim ornegi ile v. siradaki validasyon ornegi arasiu uzaklik
        en_yakin_komsular=np.argsort(uzakliklar)#egitim örneklerinin sinif merak edilenvalidasyon ornegine yakinliklarina göre siralanmasi
        en_yakin_komsular_siniflar=egitim_Y[en_yakin_komsular[:k]] #egitim örneklerinin ilk k tanesini aliyoruz
        en_cok_gorulen_sinif=stats.mode(en_yakin_komsular_siniflar)[0][0]#en yakin k egitim orneginde en cok gorulen sinif
        tahminler.append(en_cok_gorulen_sinif)
    #bu noktada tum validasyon orneklerini siniflandirmis oluyoruz, simdi bu tahminleri
    #validasyon örneklerinin gercek siniflari ile karsilastiriyoruz
    basari=0   
    for v in range(val_num):
        if tahminler[v]==val_Y[v]:#dogru tahmin ettigimiz her validasyon ornegi icin basari sayimizi bir artiriyoruz.
            basari+=1
    
    performans=(basari/val_num)*100 #dikkat edersek burda en disardaki for loop'unun icindeyiz, elde edilen bu performans  belirli bir k degeri icin elde edilen performanstir
    performanslar.append(performans)

best_k=aday_k[np.argmax(performanslar)] #np.argmax(performanslar kacinci k degerinde en yuksek performans alindigini verir

####################################
#bundan sonra validasyon setini kullanarak öğrendigimiz k degeri icin test örneklerini yine ayni validasyon setinde oldugu gibi siniflandiriyoruz.
tahminler=[]
    
for t in range(test_num):
    sinifi_merak_edilen=test_X[t,:]
    uzakliklar=[]
    for e in range(egitim_num):
        test_edilen=egitim_X[e,:]
        uzaklik=oklid_uzaklik(sinifi_merak_edilen,test_edilen)
        uzakliklar.append(uzaklik)
    en_yakin_komsular=np.argsort(uzakliklar)
    en_yakin_komsular_siniflar=egitim_Y[en_yakin_komsular[:best_k]]
    en_cok_gorulen_sinif=stats.mode(en_yakin_komsular_siniflar)[0][0]
    tahminler.append(en_cok_gorulen_sinif)
    
basari=0   
for t in range(test_num):
    if tahminler[t]==test_Y[t]:
        basari+=1
    
performans=(basari/test_num)*100     

print("k-En yakin Komsu siniflandirma performansi: {}".format(performans))

































    
    
    
    