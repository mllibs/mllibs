import numpy as np
import pandas as pd
from collections import OrderedDict
from mllibs.nlpi import nlpi

'''

Pandas DataFrame related Operations


'''


# sample module class structure
class dataframe_oper(nlpi):
    
    # called in nlpm
    def __init__(self,nlp_config):
        self.name = 'pd_df'             
        self.nlp_config = nlp_config 
        
    # called in nlpi
    def sel(self,args:dict):
        
        self.select = args['pred_task']
        self.args = args
        
        if(self.select == 'groupby'):
            self.dfgroupby(self.args)
        elif(self.select == 'concat'):
            self.dfconcat(self.args)
        elif(self.select == 'subset_concat'):
            self.subset_label(self.args)
            
    # Groupby DataFrame (or Pivot Table)
    
    def dfgroupby(self,args:dict):

        def groupby(df:pd.DataFrame,i:str,c:str=None,v:str=None,agg='mean'):
    
            if(i is not None or v is not None):
                return pd.pivot_table(data=df,
                						  index = i,
                						  columns=c,
                						  values=v,
                						  aggfunc=agg)
            else:
                return df.groupby(by=i).agg(agg)
        
        grouped_data = groupby(args['data'],
                               args['row'],
                               c=args['col'],
                               v=args['val'],
                               agg=args['agg'])
                                
        nlpi.memory_output.append(grouped_data)
                
    # Merge DataFrames

    def dfconcat(self,args:dict):
        
        def concat(lst_df,join='outer',ax=0):
            return pd.concat(lst_df,
				               join=join,
            					 axis=ax,
            					 )
            
        # merge both data frames
        merged_df = concat(args['data'],
                           join=args['join'],
                           
                           ax=eval(args['axis']))
        
        # store result
        nlpi.memory_output.append(merged_df)
        
        
    def subset_label(self,args:dict):
    
        if(type(args['data']) is list):
        
            df1 = args['data'][0]
            df2 = args['data'][1]
        
            def subset_merge(df1:pd.DataFrame,df2:pd.DataFrame):
                
                diff_1 = set(df1.columns) - set(df2.columns)
                diff_2 = set(df2.columns) - set(df1.columns)
                if(len(diff_1) != 0 and len(diff_2) == 0):
                    target = diff_1
                elif(len(diff_1) == 0 and len(diff_2) == 0):
                    target = diff_2
                elif(len(diff_1) > 1 or len(diff_2) > 1):
                    print('more than one column name missmatch!')
                elif(len(diff_1) == 0 and len(diff_2) == 0):
                    print('columns are identical')
    
                df1['set'] = 'first'
                df2['set'] = 'second'
                
                return pd.concat([df1,df2],axis=0)
    
            merged_df = subset_merge(df1,df2)
            merged_df.reset_index(inplace=True)
            nlpi.memory_output.append(merged_df)
            
'''

Corpus


'''
        
        
corpus_pda = OrderedDict({})
corpus_pda['groupby'] =  ['pandas groupby',
                          'groupby operation',
                          'group by pandas',
                          'dataframe groupby',
                          'pivot table groupby',
                          'pivot table',
                          'pivot data',
                          'pivot operation',
                          'do pivot operation']
                             
                             
corpus_pda['concat'] = ['concat dataframe',
                        'concatenate dataframe',
                        'concat df',
                        'merge dataframe',
                        'merge df',
                        'combine df',
                        'combine dataframes']
                        
corpus_pda['subset_concat'] = ['merge subsets',
								  'concat subsets',
								  'concatenate subsets',
								  'compare subset dataframes',
								  'compare subset df',
           					  'label subset dataframes',
            					  'create subset dataframe labels',
        
]
 
'''

Module Information Dictionary


'''

 
info_pda = {}
             
info_pda['groupby'] = {'module':'pd_df',
                      'action':'action',
                      'topic':'topic',
                      'subtopic':'sub topic',
                      'input_format':'pd.DataFrame',
                      'output_format':'figure',
                      'description':'pandas groupby operation, data wrangling with index, column and values'}

info_pda['concat'] = {'module':'pd_df',
                      'action':'action',
                      'topic':'topic',
                      'subtopic':'sub topic',
                      'input_format':'pd.DataFrame',
                      'output_format':'figure',
                      'description':'merge together two dataframes'}
                      
info_pda['subset_concat'] = {'module':'pd_df',
                     			 'action':'action',
                      		 'topic':'topic',
                      		 'subtopic':'sub topic',
		                      'input_format':'pd.DataFrame',
       			               'output_format':'figure',
              			        'description':'label two subset dataframes'}


configure_pda = {'corpus':corpus_pda,'info':info_pda}