import numpy as np
import pandas as pd

'''
##############################################################################

[SOURCE] NER EXTRACTION
    
##############################################################################
'''

# find all the SOURCE related tokens that need to removed and remove them
# from self.token_info uses NER tags for B-SOURCE/I-SOURCE 

# find source token 
# check x-1, x-2 tokens and if they are B-SOURCE,I-SOURCE, remove them

# use keep_token to prevent from 

def data_tokenfilter(tdf:pd.DataFrame):

    # identify source related index and remove them
    ls = tdf.copy()

    # number of data sources
    nsources = len(ls[~ls['data'].isna()])

    # either there are multiple or only a single one
    try:
        max_lendiff = int(np.max(ls[ls['dtype'].notna()]['index_id'].diff()))
    except:
        max_lendiff = 0

    # number of data sources > 0 to activate

    if(nsources > 0):

        # ONE SOURCE CASE
        
        if(max_lendiff == 0):

            # eg. in dataA
            print('[note] one source token format')

            # all index that needs to be removed
            lst_remove_idx = []
            
            # get the data index id
            p0 = list(ls[ls['dtype'].notna()]['index_id'])[0]
            lst_remove_idx.append(p0)

            ''' CHECKING PREVIOUS TOKENS '''

            # create window to check previous tokens
            pm1 = p0 - 1
            pm2 = p0 - 2

            # check if previous token belongs to SOURCE token
            try:
                source_test_pm1 = ls.loc[pm1,'ner_tags'] in ['B-SOURCE','I-SOURCE']
                source_test_pm2 = ls.loc[pm2,'ner_tags'] in ['B-SOURCE','I-SOURCE']
            except:
                source_test_pm1 = ls.loc[pm1,'ner_tags'] in ['B-SOURCE','I-SOURCE']
                source_test_pm2 = None

            if(source_test_pm2):
                lst_remove_idx.append(pm1) # remove punctuation token
                lst_remove_idx.append(pm2) # remove SOURCE token
            elif(source_test_pm1):
                lst_remove_idx.append(pm1) # remove SOURCE token
            else:
                pass # nothing needs to be removed

        elif(max_lendiff == 1):

            # eg. using dataA dataB
            print('[note] two sources tokens side by side format')
                
            # get data index id
            lst_remove_idx = list(ls[ls['dtype'].notna()]['index_id'])
            p0 = lst_remove_idx[0] # first index only

            ''' CHECKING PREVIOUS TOKENS '''

            # create window to check previous tokens
            pm1 = p0 - 1
            pm2 = p0 - 2

            # check if previous token belongs to SOURCE token
            try:
                source_test_pm1 = ls.loc[pm1,'ner_tags'] in ['B-SOURCE','I-SOURCE']
                source_test_pm2 = ls.loc[pm2,'ner_tags'] in ['B-SOURCE','I-SOURCE']
            except:
                source_test_pm1 = ls.loc[pm1,'ner_tags'] in ['B-SOURCE','I-SOURCE']
                source_test_pm2 = None

            if(source_test_pm2):
                lst_remove_idx.append(pm1) # remove punctuation token
                lst_remove_idx.append(pm2) # remove SOURCE token
            elif(source_test_pm1):
                lst_remove_idx.append(pm1) # remove SOURCE token
            else:
                pass # nothing needs to be removed

        elif(max_lendiff == 2):

            # eg. dataA and dataB
            print('[note] two sources separated by a single token format')
                
            lst_remove_idx = list(ls[ls['dtype'].notna()]['index_id'])    
            lst_remove_idx.append(lst_remove_idx[0] + 1)
            p0 = lst_remove_idx[0] # first index only

            ''' CHECKING PREVIOUS TOKENS '''

            # create window to check previous tokens
            pm1 = p0 - 1
            pm2 = p0 - 2

            # check if previous token belongs to SOURCE token
            try:
                source_test_pm1 = ls.loc[pm1,'ner_tags'] in ['B-SOURCE','I-SOURCE']
                source_test_pm2 = ls.loc[pm2,'ner_tags'] in ['B-SOURCE','I-SOURCE']
            except:
                source_test_pm1 = ls.loc[pm1,'ner_tags'] in ['B-SOURCE','I-SOURCE']
                source_test_pm2 = None

            if(source_test_pm2):
                lst_remove_idx.append(pm1) # remove punctuation token
                lst_remove_idx.append(pm2) # remove SOURCE token
            elif(source_test_pm1):
                lst_remove_idx.append(pm1) # remove SOURCE token
            else:
                pass # nothing needs to be removed

        else:
            print('[note] multiple sources w/ distance => 2 found (error)')

        '''

        Keep Tokens

        '''
        # remove indicies from [remove_idx] if condition [keep_token] is met
        
        keep_idx = list(ls[ls['keep_token'] == True].index)

        if(len(keep_idx)>0):
            lst_remove_idx = [value for index, value in enumerate(lst_remove_idx) if value not in keep_idx]

        # update token_info
        # self.mtoken_info = self.mtoken_info[~self.mtoken_info['index_id'].isin(lst_remove_idx)]
        tdf = tdf[~tdf['index_id'].isin(lst_remove_idx)]

    return tdf