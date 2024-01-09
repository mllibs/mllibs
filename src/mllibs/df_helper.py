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

'''

check if columns in dataframe contain a date/datetime format
return the list of all such column names

'''

def find_datecolumns(df:pd.DataFrame):

    # Check if each column is in a date format
    lst_date_columns = []
    for column in df.columns:
        try:
            pd.to_datetime(df[column])
            lst_date_columns.append(column)
        except:
            pass

    if(len(lst_date_columns) == 0):
        return None
    else:
        return lst_date_columns
    
'''

Convert all date like columns into datetime format

'''

def convert_datecolumns(df:pd.DataFrame):

    # Check if each column is in a date format
    for column in df.columns:
        try:
            pd.to_datetime(df[column])
            df[column] = pd.to_datetime(df[column])
        except:
            pass

    return df