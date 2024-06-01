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



def nlpilist_to_df(ddata:dict):

    '''

    Import a list of data names (from nlpi.data) to a DataFrame

    module_args['data'] -> pd.DataFrame

    '''

    data_names = ddata['data'] # list of data names

    def get_data(name:str):
        return nlpi.data[name]['data']
    
    ldata = []; pass_id = True
    for key in data_names:
        if(isinstance(get_data(key),list)):
            ldata.append(pd.Series(get_data(key), index=range(len(get_data(key))),name=key))
        else:
            print('[note] data not a list')

    print(pass_id)

    if(pass_id):
        return pd.concat(ldata,axis=1)
