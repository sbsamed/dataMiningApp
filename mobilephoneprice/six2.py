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
pricerange_np_pca= pricerange_np[2:2001:3,0:14] # 2-5-8-11-14-17.. nolu satırlar 15.sutuna kadar alındı
pricerange_np_sinif= pricerange_np[2:2001:3,14] #knn alg için sadece sınıf değerleri alında daha sonra kullnlcak



#değerlerimizi ölçeklendirdik
scaler=StandardScaler()
scaler.fit(pricerange_np_pca)
scaled_features=scaler.transform(pricerange_np_pca)
scaled_features=pd.DataFrame(scaled_features,columns=('battery_power',
'clock_speed','fc','int_memory','m_dep','mobile_wt','n_cores','pc','px_height','px_width',
'ram','sc_h','sc_w','talk_time'))
scaled_features.head()


#ölçeklendirilmiş özelliklerden en önemli iki tane özelliği aldık(PCA)
PCA=PCA(n_components=(2))
PCA.fit(scaled_features)
Transformed_PCA=PCA.transform(scaled_features)
Transformed_PCA
pca_iki_sinif=np.array(Transformed_PCA)
pca_knn = pd.DataFrame(pca_iki_sinif)
pca_knn.insert(2,"sinif",pricerange_np_sinif,True)
pca_knn=np.array(pca_knn)


X = pca_knn[:,0:2]
y = pca_knn[:,2]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))















