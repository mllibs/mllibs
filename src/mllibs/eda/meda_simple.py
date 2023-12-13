import numpy as np
import pandas as pd
from collections import OrderedDict
from mllibs.nlpi import nlpi
from mllibs.nlpm import parse_json
import pkg_resources
import json
    
'''

Simple DataFrame EDA operations

'''

class eda_simple(nlpi):
    
    def __init__(self):
        self.name = 'eda_simple'     
        self.select = None          
        self.args = None  

        path = pkg_resources.resource_filename('mllibs', '/eda/meda_simple.json')
        with open(path, 'r') as f:
            self.json_data = json.load(f)
            self.nlp_config = parse_json(self.json_data)
    
    '''

    Select relevant activation function

    '''
                    
    def sel(self,args:dict):
        
        # activation function class name
        select = args['pred_task'] 
                    
        # activate relevant function 
        if(select == 'show_info'):
            self.show_info(args)
        if(select == 'show_missing'):
            self.show_missing(args)
        if(select == 'show_stats'):
            self.show_statistics(args)
        if(select == 'show_dtypes'):
            self.show_dtypes(args)
        if(select == 'show_corr'):
            self.show_correlation(args)
        if(select == 'show_feats'):
            self.show_features(args)    
        # if(select == 'drop_column'):
        #     self.drop_column(args)        
          
    ''' 

    Module Activation Function Content

    '''
    
    @staticmethod
    def show_missing(args:dict):
        print(args['data'].isna().sum(axis=0))
        
    @staticmethod
    def show_statistics(args:dict):
        display(args['data'].describe())
        
    @staticmethod
    def show_dtypes(args:dict):
        print(args['data'].dtypes)
        
    @staticmethod
    def show_correlation(args:dict):
        corr_mat = pd.DataFrame(np.round(args['data'].corr(),2),
                             index = list(args['data'].columns),
                             columns = list(args['data'].columns))
        corr_mat = corr_mat.dropna(how='all',axis=0)
        corr_mat = corr_mat.dropna(how='all',axis=1)
        display(corr_mat)
                                      
    @staticmethod
    def show_info(args:dict):
        print(args['data'].info())

    @staticmethod
    def show_features(args:dict):
        print(args['data'].columns)

    # @staticmethod
    # def drop_column(args:dict):
    #     data_name = args['data_name'][0]
    #     nlpi.data[data_name] = 
