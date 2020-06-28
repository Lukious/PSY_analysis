'''

Coded by Lukious
My psyside project

'''

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.manifold import TSNE
from sklearn import linear_model
from sklearn.metrics import mean_squared_error



def Preprocessing(data):
    data = data.dropna()
    print(data.shape[0])
    return data

if __name__ == '__main__': 
    raw_data = pd.read_csv("./data_ori.csv")
    clr_data = Preprocessing(raw_data)

    
    data = pd.DataFrame(clr_data[['decision','decision_time']])
    data = data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]

    
    scaler = StandardScaler()
    scaler.fit(data)
    X_scaled = scaler.transform(data)
    np.mean(X_scaled), np.std(X_scaled)

    pcar = PCA()
    #pcar = PCA(n_components=2)
    pcar.fit(X_scaled)
    
    pca_d = pcar.transform(X_scaled)
    
    plt.scatter(pca_d[:,0],pca_d[:,1])
    plt.show()
    
    tsne = TSNE(random_state=0)
    digits_tsne = tsne.fit_transform(X_scaled)
    plt.scatter(digits_tsne[:,0],digits_tsne[:,1])
    plt.show()
    
    
    #RG#1
    linear_regression = linear_model.LinearRegression()
    linear_regression.fit(X=pd.DataFrame(data["decision"]), y= data["decision_time"])
    prediction = linear_regression.predict(X=pd.DataFrame(data["decision"]))

    print("a value: ", linear_regression.intercept_)
    print("b value: ", linear_regression.coef_)

    residuals = data["decision_time"] - prediction
    residuals.describe()
    
    SSE = (residuals**2).sum()
    SST = ((data["decision_time"]-data["decision_time"].mean())**2).sum()
    R_squared = 1-(SSE/SST)
    print("R_squared: ", R_squared)
    
    data.plot(kind="scatter", x = 'decision', y='decision_time', color="black")
    plt.plot(data['decision'], prediction, color='blue')
    
    print('score: ', linear_regression.score(X=pd.DataFrame(data["decision"]), y=data["decision_time"]))
    print('Mean Squared Error: ', mean_squared_error(prediction, data["decision_time"]))
    print("RMSE: ", mean_squared_error(prediction, data["decision_time"])**0.5)
    
    
    
    #RG#2
    '''
    linear_regression = linear_model.LinearRegression()
    linear_regression.fit(X=pd.DataFrame(data["decision_time"]), y= data["decision"])
    prediction = linear_regression.predict(X=pd.DataFrame(data["decision_time"]))

    print("a value: ", linear_regression.intercept_)
    print("b value: ", linear_regression.coef_)

    residuals = data["decision"] - prediction
    residuals.describe()
    
    SSE = (residuals**2).sum()
    SST = ((data["decision"]-data["decision"].mean())**2).sum()
    R_squared = 1-(SSE/SST)
    print("R_squared: ", R_squared)
    
    data.plot(kind="scatter", x = 'decision_time', y='decision', color="black")
    plt.plot(data['decision_time'], prediction, color='blue')
    
    print('score: ', linear_regression.score(X=pd.DataFrame(data["decision_time"]), y=data["decision"]))
    print('Mean Squared Error: ', mean_squared_error(prediction, data["decision"]))
    print("RMSE: ", mean_squared_error(prediction, data["decision"])**0.5)
    '''