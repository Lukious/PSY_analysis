'''

Coded by Lukious
My psyside project

'''

import pandas as pd
import numpy as np



def Preprocessing(data):
    data = data.dropna()
    print(data.shape[0])
    return data

if __name__ == '__main__': 
    raw_data = pd.read_csv("./data.csv")
    clr_data = Preprocessing(raw_data)
    
    data = pd.DataFrame(clr_data[['fast','decision','age','gender']])
    label = pd.DataFrame(clr_data['ed'])
    


    
                                  