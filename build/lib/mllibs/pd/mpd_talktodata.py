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

    # show the missing data in the column 

    def dfcolumn_na(self,args:dict):

        if(args['column'] != None):
            ls = args['data'][args['column']]
        elif(args['col'] != None):
            ls = args['data'][args['col']]
        else:
            print('[note] please specify the column name')

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