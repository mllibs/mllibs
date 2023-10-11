import pandas as pd

# data converters 

def convert_to_df(ldata):
    
    if(type(ldata) is list or type(ldata) is tuple):
        return pd.Series(ldata).to_frame()
    elif(type(ldata) is pd.Series):
        return ldata.to_frame()
    else:
        raise TypeError('Could not convert input data to dataframe')
        

def convert_to_list(ldata):
    
    if(type(ldata) is str):
        return [ldata]
    else:
        raise TypeError('Could not convert input data to list')