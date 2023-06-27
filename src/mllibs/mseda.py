import numpy as np
import pandas as pd
from collections import OrderedDict
from mllibs.nlpi import nlpi
    
'''


Simple DataFrame EDA operations


'''

class simple_eda(nlpi):
    
    def __init__(self,nlp_config):
        self.name = 'eda'             # unique module name identifier
        self.nlp_config = nlp_config  # text based info related to module
        self.select = None            # store index which module was activated
        # store all module arguments which were passed to module
        self.args = None              
    
    # describe contents of class
                    
    def sel(self,args:dict):
        
        select = args['pred_task']
                    
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
          
    ''' Module Activation Functions '''
    
    # each function needs to utilise args if they arent empty
    
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

'''

Corpus

'''
        
# Create Dataset of possible commands
corpus_eda = OrderedDict({
    
                'show_info' : ['show data information',
                               'show dataset information',
                               'show dataframe information',
                               'show dataframe info',
                               'data info',
                               'df info',
                               'dataframe information',
                               'data info'],
    
    
              "show_missing":['find missing data',
                              'missing data.',
                              'data missing',
                              'show missing data',
                              'show missing data in columns',
                              'show column missing data',
                              'column missing data',
                              'show data which is missing',
                              'print missing data'],

            'show_stats': ['show describe',
                           'show statistics',
                           'show stats'
                           'show dataframe statistics',
                           'statistics',
                           'statistics minimum',
                           'statistics maximum',
                           'statistics mean',
                           'stats minimum',
                           'stats maximum',
                           'stats mean',
                           'pandas describe',
                           'pandas statistics',
                           'pandas stats'
                           'statistics dataframe',
                           'dataframe statistics'  
                           'tabular statistics'],
            
            'show_dtypes': ['show dataframe dtypes',
                            'show data types',
                            'print data types',
                            'print dtypes',
                            'show feature dtypes',
                            'show dtype',
                            'feature types',
                            'dtype',
                            'dtypes'],
            
            'show_corr': ['create correlation matrix',
                          'plot correlation matrix',
                          'generate correlation matrix',
                          'create correlation',
                          'correlation',
                          'show feature correlation',
                          'show correlation',
                          'genenerate correlation matrix',
                          'show correlation matrix',
                          'show correlation of features'],

})


# Other useful information about the task
info_eda = {'show_info':{'module':'eda',
                               'action':'table operation',
                               'topic':'exploratory data analysis (eda)',
                               'subtopic':'inspect dataframe',
                               'input_format':'pd.DataFrame',
                               'description':'show dataframe information'},
    
            'show_missing':{'module':'eda',
                            'action':'show plot',
                            'topic':'exploratory data analysis (eda)',
                            'subtopic':'visualise missing data',
                            'input_format':'pd.DataFrame',
                            'description':'show showing missing data in data frame columns'},
                  
                 'show_stats':{'module':'eda',
                               'action':'table operation',
                               'topic':'exploratory data analysis (eda)',
                               'subtopic':'inspect dataframe',
                               'input_format':'pd.DataFrame',
                               'description':'show statistics of numerical columns in dataframe using describe'},
                 
             'show_dtypes': {'module':'eda',
                             'action':'table operation',
                             'topic':'exploratory data analysis (eda)',
                             'subtopic':'inspect dataframe',
                             'input_format':'pd.DataFrame',
                             'description':'show column data types dtypes'},
                  
                 'show_corr': {'module':'eda',
                               'action':'show plot',
                               'topic':'exploratory data analysis (eda)',
                               'subtopic':'show correlation',
                               'input_format':'pd.DataFrame',
                               'description':'visualise linear correlation between features'},
            
                 }

configure_eda = {'corpus':corpus_eda,'info':info_eda}