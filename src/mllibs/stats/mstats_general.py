
from mllibs.nlpi import nlpi
from mllibs.dict_helper import sfp,sfpne
import pandas as pd
import numpy as np
from collections import OrderedDict
import warnings; warnings.filterwarnings('ignore')
from mllibs.nlpm import parse_json
import pkg_resources
import json
import seaborn as sns
from mllibs.module_helper import confim_dtype
from mllibs.data_conversion import nlpilist_to_df
import textwrap

'''

Visualise Statistical Differences

'''

class stats_general(nlpi):
    
    def __init__(self):

        self.name = 'stats_general'  
        path = pkg_resources.resource_filename('mllibs','/stats/mstats_general.json')
        with open(path, 'r') as f:
            self.json_data = json.load(f)
            self.nlp_config = parse_json(self.json_data)    

    '''
    ////////////////////////////////////////////////////////////

                    Select Activation Function

    ////////////////////////////////////////////////////////////
    '''
        
    def sel(self,args:dict):

        self.args = args
        select = args['pred_task']
        self.data_name = args['data_name']
        self.subset = args['subset']
        self.info = args['task_info']['description']
        
        # check_dtype_id = confim_dtype(self.args['dtype_req'],self.args['ldata'])

        if(nlpi.silent == False):
            print('\n[note] module function info');print(textwrap.fill(self.info, 60));print('')

        
        if(select == 'gstat_stats'):

            '''

            DataFrame/List Statistics 

            '''

            # only lists are mentioned
            if(args['sub_task'] == 'list_inputs'):

                lst_data = []
                for name in args['data']['list']:
                    ldata = nlpi.data[name]['data']
                    data = pd.DataFrame(ldata,columns=['data'])
                    data['sample'] = name
                    data['unique_id'] = range(len(data))
                    lst_data.append(data)

                combined = pd.concat(lst_data)
                combined = combined.reset_index(drop=True)
                df_pivot = combined.pivot(index='unique_id',columns='sample', values='data')
                args['data'] = df_pivot
                self.gs_allstats(args)

            elif(args['sub_task'] == 'dataframe_inputs'):

                '''

                When dataframe is the input sources only 

                '''

                lst_data = []
                for name in args['data']['df']:
                    ldata = nlpi.data[name]['data']
                    lst_data.append(ldata)

                args['data'] = lst_data[0]

                if(len(lst_data) == 1):
                    self.gs_allstats(args)
                else:
                    print('[note] please use only one datafame')

            elif(args['sub_task'] == 'dataframe_subset'):

                '''

                When dataframe and columns are specified in request

                '''

                lst_data = []
                for name in args['data']['df']:
                    ldata = nlpi.data[name]['data']
                    lst_data.append(ldata)

                args['data'] = lst_data[0][args['column']]

                if(len(lst_data) == 1):
                    self.gs_allstats(args)
                else:
                    print('[note] please use only one datafame')

    '''
    ////////////////////////////////////////////////////////////

                       Activation Functions

    ////////////////////////////////////////////////////////////
    '''

    def gs_allstats(self,args:dict):
        display(args['data'].describe())
