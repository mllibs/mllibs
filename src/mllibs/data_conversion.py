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

'''

Convert list to:

        - [series] defaults to list name /w rename option  
        - [dataframe]     '' 
        - [dictionary]    ''

'''

def convert_list(data:list,output_type:str,name:str=None):
    
    # series
    if output_type == 'series':
        if(name == None):
            return pd.Series(data,name='list')
        else:
            return pd.Series(data,name=f'{list}')

    # dataframe
    elif output_type == 'dataframe':
        if(name == None):
            return pd.DataFrame(data,columns=[f'{name}'])
        else:
            return pd.DataFrame(data,columns=['list'])

    # dictionary
    elif output_type == 'dictionary':
        if(name == None):
            return {'list':data}
        else:
            return {f'{name}':data}
    else:
        return "Invalid output type"


'''

Convert pandas series to:

        - [list_data] : value list
        - [list_index] : value index
        - [dataframe] : dataframe w/ rename option
        - [dict_index] : dictionary {index : value}
        - [dict_rindex] : dictionary {value : index}
        - [dict_name] : dictionary {sname : values} w/ rename option

'''

def convert_series(data:pd.Series,output_type:str,name:str=None):

    # list
    if(output_type == 'list_data'):
        return data.tolist()
    elif(output_type == 'list_index'):
        return list(data.index)

    # dataframe
    elif(output_type == 'dataframe'):
        if(name == None):
            return data.to_frame()
        else:
            ldf = data.to_frame()
            ldf.columns = [f'{name}']
            return ldf

    # dictionary
    elif(output_type == 'dict_index'):
        return data.to_dict()
    elif(output_type == 'dict_rindex'):
        return {v: k for k, v in data.to_dict().items()}
    elif(output_type == 'dict_name'):
        if(name == None):
            return {data.name:list(data.values)}
        else:
            return {f'{name}':list(data.values)}