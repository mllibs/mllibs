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

    @staticmethod
    def sfp(args,preset,key:str):
    
        if(args[key] is not None):
            return args[key]
        else:
            return preset[key] 
        
    # called in nlpi
    def sel(self,args:dict):
        
        self.select = args['pred_task']
        self.args = args
        
        if(self.select == 'dfcolumninfo'):
            self.dfgroupby(self.args)
        if(self.select == 'dfsize'):
            self.dfsize(self.args)
        if(self.select == 'dfcolumn_distr'):
            self.dfcolumn_distr(self.args)
        if(self.select == 'dfcolumn_na'):
            self.dfcolumn_na(self.args)
        if(self.select == 'dfall_na'):
            self.dfall_na(self.args)
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
            self.dfcolumn_unique(self.args)

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
            display(args['data'][args['column']].value_counts())
        elif(args['col'] != None):
            display(args['data'][args['col']].value_counts())
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


    # show the missing data in the column 

    def dfcolumn_na(self,args:dict):

        if(args['column'] != None):
            ls = args['data'][args['column']]
        elif(args['col'] != None):
            ls = args['data'][args['col']]
        else:
            print('[note] please specify the column name')
            ls = None

        if(ls != None):

            # convert series to dataframe
            if(isinstance(ls,pd.DataFrame) == False):
                ls = ls.to_frame()

            print("[note] I've stored the missing rows")
            nlpi.memory_output.append({'data':ls[ls.isna().any(axis=1)]})            

    # show the missing data in all columns

    def dfall_na(self,args:dict):
        
        print(args['data'].isna().sum().sum(),'rows in total have missing data')
        print(args['data'].isna().sum())

        print("[note] I've stored the missing rows")
        ls = args['data']
        nlpi.memory_output.append({'data':ls[ls.isna().any(axis=1)]})  

    # show dataframe statistics  

    @staticmethod
    def show_statistics(args:dict):
        display(args['data'].describe())

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
        display(corr_mat)

