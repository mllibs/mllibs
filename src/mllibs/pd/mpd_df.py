from mllibs.nlpi import nlpi
import pandas as pd
import warnings; warnings.filterwarnings('ignore')
from mllibs.nlpm import parse_json
import pkg_resources
import json

'''

Pandas DataFrame related Operations


'''

# sample module class structure
class pd_df(nlpi):
    
    def __init__(self):
        self.name = 'pd_df'             
        path = pkg_resources.resource_filename('mllibs','/pd/mpd_df.json')
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
        
        if(self.select == 'dfgroupby'):
            self.dfgroupby(self.args)
        elif(self.select == 'dfconcat'):
            self.dfconcat(self.args)
        elif(self.select == 'subset_concat'):
            self.subset_label(self.args)
            
    ''' 
    
    ACTIVATION FUNCTIONS 

    '''

    # Groupby DataFrame (or Pivot Table)
    
    def dfgroupby(self,args:dict):

        # preset values
        pre = {'agg':'mean'}

        # groupby helper function
        
        def groupby(df:pd.DataFrame, # input dataframe
                    i:str,           # index
                    c:str=None,      # column
                    v:str=None,      # value
                    agg='mean'       # aggregation function
                    ):
    
            # pivot table / standard groupby
            if(i is not None or v is not None):
                return pd.pivot_table(data=df,
                                      index = i,
                                      columns=c,
                                      values=v,
                                      aggfunc=agg)
            else:
                return df.groupby(by=i).agg(agg)
        
        # general groupby function (either pivot_table or groupby)
        grouped_data = groupby(args['data'],
                               args['row'],
                               c=args['col'],
                               v=args['val'],
                               agg=self.sfp(args,pre,'agg'))
           
        nlpi.memory_output.append({'data':grouped_data})
                
    # Concatenate DataFrames

    def dfconcat(self,args:dict):

        # default parameters
        pre = {'axis':0}
        
        def concat(lst_df,join='outer',ax=0):
            return pd.concat(lst_df,
				             join=join,
            				 axis=ax,
            				)
            
        # merge both data frames
        merged_df = concat(args['data'],
                           join=args['join'],
                           ax=self.sfp(args,pre,'axis'))
        
        # store result
        nlpi.memory_output.append({'data':merged_df})
        
    # Add subset label for two dataframes
        
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
            nlpi.memory_output.append({'data':merged_df})
            