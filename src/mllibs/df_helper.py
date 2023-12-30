import pandas as pd

'''

Split datatype into numeric and categorical

'''

def split_types(df:pd.DataFrame):
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']  
    numeric = df.select_dtypes(include=numerics)
    categorical = df.select_dtypes(exclude=numerics)
    return list(numeric.columns),list(categorical.columns)


'''

Check if dataframe contains the list of column names provided

'''

def check_list_in_col(df:pd.DataFrame,lst:list):

    # Get the column names of the DataFrame
    df_column_names = df.columns.tolist()
    
    # Check if all items in column_names_list are present in df_column_names
    if all(column in df_column_names for column in lst):
        return True
    else:
        return False