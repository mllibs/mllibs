from mllibs.nlpi import nlpi
import pandas as pd
import warnings; warnings.filterwarnings('ignore')
from mllibs.nlpm import parse_json
import pkg_resources
import json

'''

Data Exploration via Natural Language

'''

# sample module class structure
class pd_talktodata(nlpi):
    
    def __init__(self):
        self.name = 'pd_talktodata'             
        path = pkg_resources.resource_filename('mllibs','/pd/mpd_talktodata.json')
        with open(path, 'r') as f:
            self.json_data = json.load(f)
            self.nlp_config = parse_json(self.json_data)

    # set preset value from dictionary
    # if argument is already set
        
    # called in nlpi
    def sel(self,args:dict):
        
        self.select = args['pred_task']
        
        if(self.select == 'dfcolumninfo'):
            self.dfgroupby(args)
        if(self.select == 'dfsize'):
            self.dfsize(args)
        if(self.select == 'dfcolumn_distr'):
            self.dfcolumn_distr(args)

        if(self.select == 'dfna_column'):
            self.dfna_column(args)
        if(self.select == 'dfna_all'):
            self.dfna_all(args)
        if(self.select == 'dfna_perc'):
            self.dfna_perc(args)
            
        # convert column types
            
        if(self.select == 'dfcolumn_tostr'):
            self.dfcolumn_tostr(args)
        if(self.select == 'dfcolumn_toint'):
            self.dfcolumn_toint(args)
        if(self.select == 'dfcolumn_dtype'):
            self.dfcolumn_dtype(args)

        if(self.select == 'show_stats'):
            self.show_statistics(args)
        if(self.select == 'show_info'):
            self.show_info(args)
        if(self.select == 'show_dtypes'):
            self.show_dtypes(args)
        if(self.select == 'show_feats'):
            self.show_features(args)   
        if(self.select == 'show_corr'):
            self.show_correlation(args)
        if(self.select == 'dfcolumn_unique'):
            self.dfcolumn_unique(args)
        if(self.select == 'df_preview'):
            self.df_preview(args)

    ''' 
    
    ACTIVATION FUNCTIONS 

    '''

    # show dataframe columns
    
    def dfcolumninfo(self,args:dict):
        print(args['data'].columns)

    # show size of dataframe

    def dfsize(self,args:dict):
        print(args['data'].shape)

    # column distribution

    def dfcolumn_distr(self,args:dict):

        if(args['column'] != None):
            display(args['data'][args['column']].value_counts(dropna=False))
        elif(args['col'] != None):
            display(args['data'][args['col']].value_counts(dropna=False))
        else:
            print('[note] please specify the column name')

    # column unique values

    def dfcolumn_unique(self,args:dict):

        if(args['column'] == None and args['col'] == None):
            print('[note] please specify the column name')
        else:
            if(args['column'] != None):
                print(args['data'][args['column']].unique())
            elif(args['col'] != None):
                print(args['data'][args['col']].unique())

    # show the missing data in the column / if no column is provided show for all columns

    def dfna_column(self,args:dict):

        if(args['column'] != None):
            ls = args['data'][args['column']]
        elif(args['col'] != None):
            ls = args['data'][args['col']]
        else:
            print('[note] please specify the column name, showing for all columns')
            ls = args['data']

        if(isinstance(ls,pd.Series) == True):
            tls = ls.to_frame()
            print(tls.isna().sum().sum(),'rows in total have missing data')
            print("[note] I've stored the missing rows")
            idx = tls[tls.isna().any(axis=1)].index
            nlpi.memory_output.append({'data':args['data'].loc[idx]})          
        elif(isinstance(ls,pd.DataFrame) == True):
            print(args['data'].isna().sum().sum(),'rows in total have missing data')
            print(args['data'].isna().sum())
            print("[note] I've stored the missing rows")
            nlpi.memory_output.append({'data':ls[ls.isna().any(axis=1)]})            

    # show the missing data in all columns

    def dfna_all(self,args:dict):
        
        print(args['data'].isna().sum().sum(),'rows in total have missing data')
        print(args['data'].isna().sum())

        print("[note] I've also stored the missing rows!")
        ls = args['data']
        nlpi.memory_output.append({'data':ls[ls.isna().any(axis=1)]})  

    @staticmethod
    def dfna_perc(args:dict):

        na_percentage = args['data'].isna().mean() * 100
        try:
            display(na_percentage)
        except:
            print(na_percentage)

        print("[note] I've also stored the missing rows!")
        ls = args['data']
        nlpi.memory_output.append({'data':ls[ls.isna().any(axis=1)]})  
        

    '''
    
    convert column types
    
    '''
        
    def dfcolumn_dtype(self,args:dict):
        
        # parameter dtype needs to have been set
        if(args['dtype'] != None):
            
            data = nlpi.data[args['data_name']]['data']
            column = args['column']
            
            try:
                data[column] = data[column].astype(args['dtype'])
                print('[note] modifying original dataset!')
            except:
                print(f"[note] can't modify the existing column type to {args['dtype']}")
        
    # convert column to string
    
    def dfcolumn_tostr(self,args:dict):
        
        data = nlpi.data[args['data_name']]['data']
        column = args['column']
        
        try:
            data[column] = data[column].astype('string')
            print('[note] modifying original dataset!')
        except:
            print("[note] can't modify the existing column type to string!")
        
    # convert column type to integer
    
    def dfcolumn_toint(self,args:dict):
        
        data = nlpi.data[args['data_name']]['data']
        column = args['column']
        
        try:
            data[column] = data[column].astype('int')
            print('[note] modifying original dataset!')
        except:
            print("[note] can't modify the existing column type to integer!")
        
    # show dataframe statistics
    
    @staticmethod
    def show_statistics(args:dict):
        try:
            display(args['data'].describe())
        except:
            print(args['data'].describe())

    # show dataframe information

    @staticmethod
    def show_info(args:dict):
        print(args['data'].info())

    # show dataframe column data types

    @staticmethod
    def show_dtypes(args:dict):
        print(args['data'].dtypes)

    # show column features

    @staticmethod
    def show_features(args:dict):
        print(args['data'].columns)

    # show numerical column linear correlation in dataframe

    @staticmethod
    def show_correlation(args:dict):
        corr_mat = pd.DataFrame(np.round(args['data'].corr(),2),
                             index = list(args['data'].columns),
                             columns = list(args['data'].columns))
        corr_mat = corr_mat.dropna(how='all',axis=0)
        corr_mat = corr_mat.dropna(how='all',axis=1)

        try:
            display(corr_mat)
        except:
            print(corr_mat)

    @staticmethod
    def df_preview(args:dict):
        try:
            display(args['data'].head())
        except:
            print(args['data'].head())

            