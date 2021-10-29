# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 14:33:48 2021

@author: Samed
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy

   
pricerange=pd.read_excel("mobilephoneprice.xlsx")#verisetimizi aldık 
pricerange_np=np.array(pricerange) #np formuna dönüştürüldü
pricerange_np= pricerange_np[2:2001:3,:] # 2-5-8-11-14-17....... nolu satırlar alındı
satir_sayisi=pricerange_np.shape[0]
sutun_sayisi=pricerange_np.shape[1]
pricerange_np=pricerange_np[np.random.permutation(satir_sayisi),:] #veriseti karıştırıldı


def sinifozelliksaybul(col): #kategorik özellikleri sınıf türlerine göre 2x4'lük bir matrise alıyoruz
    sinifozellikiliskisisayisi=np.array([[0,0,0,0,0],[0,0,0,0,0]]) 
    sinifsayisi=np.array([0,0,0,0]) 
    for row in range(satir_sayisi): #ilgili kolonda hangi özellikten(1veya0) kaç adet oldugunu bulduk
        ozellik=pricerange_np[row,col]
        sinif=pricerange_np[row,19];
        if(ozellik==1):
            sinifozellikiliskisisayisi[1,4]+=1
        elif(ozellik==0):
            sinifozellikiliskisisayisi[0,4]+=1
    for row in range(satir_sayisi):    #bir sınıfta  kategorik özelliklerdeki değerlerden kaç tane old. bulduk
        ozellik=pricerange_np[row,col]
        sinif=pricerange_np[row,19]
        if(ozellik==0)and(sinif==0):
            sinifozellikiliskisisayisi[0,0]+=1
            sinifsayisi[0]+=1                        
        elif(ozellik==0)and(sinif==1):
            sinifozellikiliskisisayisi[0,1]+=1
            sinifsayisi[1]+=1
        elif(ozellik==0)and(sinif==2):
            sinifozellikiliskisisayisi[0,2]+=1
            sinifsayisi[2]+=1
        elif(ozellik==0)and(sinif==3):
            sinifozellikiliskisisayisi[0,3]+=1
            sinifsayisi[3]+=1
        elif(ozellik==1)and(sinif==0):
            sinifozellikiliskisisayisi[1,0]+=1
            sinifsayisi[0]+=1
        elif(ozellik==1)and(sinif==1):
            sinifozellikiliskisisayisi[1,1]+=1 
            sinifsayisi[1]+=1
        elif(ozellik==1)and(sinif==2):
            sinifozellikiliskisisayisi[1,2]+=1
            sinifsayisi[2]+=1
        elif(ozellik==1)and(sinif==3):
            sinifozellikiliskisisayisi[1,3]+=1
            sinifsayisi[3]+=1
    return sinifozellikiliskisisayisi


sistementropysi=entropy([168/666,173/666,174/666,151/666,],base=2)

def entropyhesaplama(col): #burada informatian gain için kolonlara özgü entropiler hesaplandı
    ozelliklerinentropisi=np.array([0.0,0.0])
    sinifbilgileri=np.array(sinifozelliksaybul(col))
    for x in range(2):
        ozelliklerinentropisi[x]=(entropy([ sinifbilgileri[x,0]/sinifbilgileri[x,4],
        sinifbilgileri[x,1]/sinifbilgileri[x,4],
        sinifbilgileri[x,2]/sinifbilgileri[x,4],
        sinifbilgileri[x,3]/sinifbilgileri[x,4] ],base=2))
    return (ozelliklerinentropisi)
def informationgainhesapla(col): #ağırlıklı entropiyi ve bilgi kazancını burada hesapladık
    agirliklientropi=0.0
    entropibilgileri=np.array(entropyhesaplama(col))
    sinifbilgileri=np.array(sinifozelliksaybul(col))
    for i in range(2):
        agirliklientropi+=entropibilgileri[i]*sinifbilgileri[i,4]/666    
    bilgikazanci=sistementropysi-agirliklientropi
    
    print("%2d.Kolon için bilgi kazancı: %5.15f"%(col, bilgikazanci))
    print("----------------------------")
    return bilgikazanci
#######################################################################################################
kategorikozellikler=np.array([1,3,5,17,18])
print("Bilgi Kazançları:")
for x in range(5):
    informationgainhesapla(kategorikozellikler[x])
    
def kolonlarinentropisi(col): #kolonların entropisi hesaplandı
    sinifbilgileri=np.array(sinifozelliksaybul(col))
    kolonentropisi=entropy( [ sinifbilgileri[0,4]/666,sinifbilgileri[1,4]/666 ],base=2)
    return kolonentropisi
print("***************************")
print("Kolon entropileri:")
for x in range(5):
    kolonentropisi=kolonlarinentropisi(kategorikozellikler[x])
    print("%2d.Kolon entropi değeri: %5.15f"%(kategorikozellikler[x], kolonentropisi))
    print("----------------------------")

    





