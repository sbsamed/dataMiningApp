# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 18:40:52 2021

@author: Samed
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy

#verisetimizi aldık    
pricerange=pd.read_excel("mobilephoneprice.xlsx")
pricerange_np=np.array(pricerange) #np formuna dönüştürüldü
pricerange_np= pricerange_np[2:2001:3,:] # 2-5-8-11-14-17....... nolu satırlar alındı
pricerange_np=np.delete(pricerange_np, [1,3,5,17,18],1)#burada indsleri belirtilen kategorik tiptekı kolonlar silindi
satir_sayisi=pricerange_np.shape[0]
sutun_sayisi=pricerange_np.shape[1]


####SAMED BASKIN####
s_sapma=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,])
aritmetikortalama=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,])
def s_sapmabul():
    for x in range(sutun_sayisi-1):#sınıf bılgılerının oldugu kolonu dahıl etmedık
       s_sapma[x]=np.std(pricerange_np[:,x])

def aritmetikortbul():
    for x in range(sutun_sayisi-1):#sınıf bılgılerının oldugu kolonu dahıl etmedık
       aritmetikortalama[x]=np.mean(pricerange_np[:,x])
  
s_sapmabul()
aritmetikortbul()


degistirilendegerler = np.empty((0,4), float)
for y in range(sutun_sayisi-1):
    a1=aritmetikortalama[y]+(2*s_sapma[y]) 
    a2=aritmetikortalama[y]-(2*s_sapma[y])
    for x in range(satir_sayisi):
        deger=pricerange_np[x,y]
        if(deger>a1)or(deger<a2):
            pricerange_np[x,y]=aritmetikortalama[y]
            degistirilendegerler = np.append(degistirilendegerler, np.array([[x,y,deger,aritmetikortalama[y]]]), axis=0)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        