import pandas as pd
from mllibs.list_helper import get_bracket_content

'''
##############################################################################

                        [  ACTIVE COLUMNS  ]
                     
                    Extraction of active column (s)
               Active columns are defined by { brackets 

- Identification of bracket indicies
- Finding the nearest B-PARAM ner tag (to the left)
- Removing all in between these indicies, starting from B-PARAM

If columns { ac_name } > stored in self.module_args['subset']

##############################################################################
'''

def ac_extraction(tdf:pd.DataFrame, nlpi_data: dict):
    
    ls = tdf.copy()
    lst = list(ls['token'])
    module_args = {}
    
    # find all bracket pairs in list
    pairs = get_bracket_content(lst)
    
    lst_act_functions = []
    for pair in pairs:
        select_col_content = ls.iloc[pair[0]:pair[1]+1]
        act_funct = select_col_content[~select_col_content['token'].isin(['{','}'])]
        lst_act_functions.append(list(act_funct['token'])[0])
        
    # FIND WHICH AC TO STORE
        
    # find what the active columns were assigned to by selecting the first 
    # for each pair select the closest param [ner_tag] to the left!
        
    # indicies (lst_param_idx) of PARAM/SUBSET tokens
    lst_param_idx = []; lst_param_ner_tag = []; subset_id = []
    for pair in pairs:
        
        ldf = ls[(ls.index < pair[0]) & (ls['ner_tags'].isin(['B-PARAM','B-SUBSET']))].reset_index(drop=True)
        
        # if at least one match
        if(ldf.shape[0] > 0):
            
            previous_token_idx = ldf.iloc[-1]['index_id']
            previous_token_ner_tag = ldf.iloc[-1]['ner_tags']
            lst_param_idx.append(previous_token_idx)
            
            if (previous_token_ner_tag == 'B-SUBSET'):
                subset_id.append(True)
            else:
                subset_id.append(False)
                
    # select everything in between [ner_tag] and last BRACKET
    remove_idx = []
    for ii,pair in enumerate(pairs):
        if(subset_id[ii] is False):
            remove_idx.append(list(ls.iloc[lst_param_idx[ii]:pair[1]+1]['index_id']))
        else:
            remove_idx.append(list(ls.iloc[pair[0]:pair[1]+1]['index_id']))
            
    # store the TOKEN of lst_param_idx
    lst_param_id = []
    for i in lst_param_idx:
        lst_param_id.append(ls.iloc[i]['token'])
    
#   for pair in remove_idx:
#       ls = ls[~(ls.index_id.isin(pair))]
#   ls = ls.reset_index(drop=True)
#   ls['index_id'] = list(ls.index)
    
    '''

    Find all active column names (keys) (this is needed to store AC)
    
    '''
    
    # ac_data [for each dataframe data (keys) display its active colum names (values)]
    
    ac_data = {}
    for data_name in nlpi_data.keys():
        if(isinstance(nlpi_data[data_name]['data'],pd.DataFrame)):
            ac_data[data_name] = list(nlpi_data[data_name]['ac'].keys())
            
    # find_ac [given a active column name, finds its data source]
            
    def find_ac(name):
        data_name = None
        for key,values in ac_data.items():
            if(name in values):
                data_name = key
                
        if(data_name is not None):
            return data_name
        else:
            return None
        
    '''
    
    Store names into PARAMETER / SUBSET (columns,column) 
    
    '''
        
    # d_id - {d_id}
    # ac_id - param_id token identifier
        
    for d_id,ac_id in zip(lst_act_functions,lst_param_id):
        
        # get the data name of the active column
        data_name = find_ac(d_id)
        
        if(data_name != None):
            print(f'[note] storing active columns for [{ac_id}] in module_args')
            module_args[ac_id] = nlpi_data[data_name]['ac'][d_id]
            
    return module_args, ls