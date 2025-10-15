from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler

def Airline(airlinename):
    df = pd.read_csv(f"{airlinename}")
    #Year,Month,Stockprice omitted
    x = df.iloc[:,3:]
    #standardize
    standard = StandardScaler()
    x = standard.fit_transform(x)
    #Index
    pca = PCA(n_components=1)
    pca.fit(x)
    sensitivity_index = pca.transform(x)
    