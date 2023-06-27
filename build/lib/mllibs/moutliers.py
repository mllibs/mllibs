import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from copy import deepcopy
from mllibs.nlpi import nlpi

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

palette = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252']
palette_rgb = [hex_to_rgb(x) for x in palette]

'''

FIND OUTLIERS IN DATA

'''

class data_outliers(nlpi):
    
    # called in nlpm
    def __init__(self,nlp_config):
        self.name = 'outliers'          
        self.nlp_config = nlp_config  
        
    # called in nlpi

    def sel(self,args:dict):
                  
        select = args['pred_task']
        self.data_name = args['data_name']  # name of the data

        if(select == 'outlier_iqr'):
            self.outlier_iqr(args)
        elif(select == 'outlier_zscore'):
            self.outlier_zscore(args)
        elif(select == 'outlier_norm'):
            self.outlier_normal(args)
        elif(select == 'outlier_dbscan'):
            self.outlier_dbscan(args)

    @staticmethod
    def sfp(args,preset,key:str):
        
        if(args[key] is not None):
            return eval(args[key])
        else:
            return preset[key] 
        
    # set general parameter
        
    @staticmethod
    def sgp(args,key:str):
        
        if(args[key] is not None):
            return eval(args[key])
        else:
            return None

    @staticmethod
    def split_types(df):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']  
        numeric = df.select_dtypes(include=numerics)
        categorical = df.select_dtypes(exclude=numerics)
        return numeric,categorical
    
    ''' 
    
    ACTIVATION FUNCTIONS 
    
    '''
    # outlier_iqr
    # outlier_zscore
    # outlier_normal
    # outlier_dbscan
            
    # find outliers using IQR values
        
    def outlier_iqr(self,args:dict):
        
        pre = {'scale':1.5}

        df = args['data']
        scale = self.sfp(args,pre,'scale')

        # helper function
        def get_iqroutlier(df):

            dict_outlier_index = {}
            for _, v in df.items():
                q1 = v.quantile(0.25);
                q3 = v.quantile(0.75);
                irq = q3 - q1

                # select data
                v_col = v[(v <= q1 - scale * irq) | (v >= q3 + scale * irq)]
                dict_outlier_index[v_col.name] = list(v_col.index)

            return dict_outlier_index
        
        # return dictionary containing indicies of outliers
        dict_outlier_index = get_iqroutlier(df)
        
        def label_outliers(data):
            ldata = data.copy()
            ldata.loc[:,'outlier_iqr'] = 'internal'
            for _,v in data.items():
                if(len(dict_outlier_index[v.name]) != 0):
                    ldata.loc[dict_outlier_index[v.name],'outlier_iqr'] = v.name
            return ldata
        
        ldata = label_outliers(df)

         # store relevant data about operation in data source
        nlpi.memory_output.append({'data':ldata})

         # store relevant data about operation in data source
        nlpi.data[self.data_name[0]]['outlier'][f'outlier_iqr_{nlpi.iter}'] = ldata['outlier_iqr']
        
    # find outliers using z_scores

    def outlier_zscore(self,args:dict):

        pre = {'threshold':3}
        
        df = args['data']
        num,_ = self.split_types(args['data'])
        
        def outliers_z_score(ys, threshold):
            mean_y = np.mean(ys)
            std_y = np.std(ys)
            z_scores = [(y - mean_y) / std_y for y in ys]
            return np.where(np.abs(z_scores) > threshold)[0]
    
        dict_outlier_index = {}
        for _, v in num.items():
            dict_outlier_index[v.name] = list(outliers_z_score(v,self.sfp(args,pre,'threshold')))
            
        def label_outliers(data):
            ldata = data.copy()
            ldata.loc[:,'outlier_zscore'] = 'internal'
            for _,v in data.items():
                if(len(dict_outlier_index[v.name]) != 0):
                    ldata.loc[dict_outlier_index[v.name],'outlier_zscore'] = v.name
            return ldata
        
        ldata = label_outliers(df)

         # store relevant data about operation in data source
        nlpi.memory_output.append({'data':ldata})

        # store relevant data about operation in data source
        nlpi.data[self.data_name[0]]['outlier'][f'outlier_zscore_{nlpi.iter}'] = ldata['outlier_zscore']
     
    # find outliers using normal distribution
    
    def outlier_normal(self,args:dict):

        pre = {'threshold':0.014}
 
        def estimate_gaussian(dataset):
            mu = np.mean(dataset, axis=0)
            sigma = np.cov(dataset.T)
            return mu, sigma

        def get_gaussian(mu, sigma):
            distribution = norm(mu, sigma)
            return distribution

        def get_probs(distribution, dataset):
            return distribution.pdf(dataset)
        
        # loop through all columns 
        df = args['data']
   
        dict_outlier_index = {}
        for _, v in df.items():
            
            # standardisation of columns
            w = v.to_frame()
            w_sc = StandardScaler().fit_transform(w)        
            v = pd.Series(w_sc[:,0],name=v.name)
            mu, sigma = estimate_gaussian(v.dropna())
            distribution = get_gaussian(mu, sigma)

            # calculate probability of the point appearing
            probabilities = get_probs(distribution,v.dropna())        
            dict_outlier_index[v.name] = np.where(probabilities < self.sfp(args,pre,'threshold'))[0]
            
        def label_outliers(data):
            ldata = data.copy()
            ldata.loc[:,'outlier_normal'] = 'internal'
            for _,v in data.items():
                if(len(dict_outlier_index[v.name]) != 0):
                    ldata.loc[dict_outlier_index[v.name],'outlier_normal'] = v.name
            return ldata
        
        ldata = label_outliers(df)

        # store relevant data about operation in data source
        nlpi.memory_output.append({'data':ldata})

        # store relevant data about operation in data source
        nlpi.data[self.data_name[0]]['outlier'][f'outlier_normal_{nlpi.iter}'] = ldata['outlier_normal']
        
    # find outliers using dbscan
        
    def outlier_dbscan(self,args:dict):

        pre = {'eps':2.4,'min_samples':3}
            
        df = args['data']
        num,_ = self.split_types(args['data'])
   
        w_sc = StandardScaler().fit_transform(num)        
        v = pd.DataFrame(w_sc,
                         columns = num.columns, 
                         index = num.index)
        
        db = DBSCAN(eps=self.sfp(args,pre,'eps'),
                    min_samples=self.sfp(args,pre,'min_samples')).fit(v)
        
        # for all features the same
        dict_outlier_index = OrderedDict({new_list: np.where(db.labels_ == -1)[0] for new_list in list(num.columns)})
        first = dict_outlier_index[[*dict_outlier_index.keys()][0]]

        def label_outliers(data):
            ldata = data.copy()
            ldata.loc[:,'outlier_dbscan'] = 'internal'
            ldata.loc[first,'outlier_dbscan'] = 'outlier'
            return ldata
            
        ldata = label_outliers(df)

        # store relevant data about operation in data source
        nlpi.memory_output.append({'data':ldata,
                                   'model':db})

        # store relevant data about operation in data source
        nlpi.data[self.data_name[0]]['outlier'][f'outlier_dbscan_{nlpi.iter}'] = ldata['outlier_dbscan']

                    
'''

Corpus

'''
    
corpus_outliers = OrderedDict({"outlier_iqr":['find outliers in data using IQR',
                                           'find outliers using IQR',
                                           'get IQR outliers',
                                           'find IQR outliers',
                                           'find IQR outlier',
                                           'get outliers using IRQ',
                                           'inter quartile range outliers',
                                           'boxplot outliers',
                                           'get boxplot outliers'],
                            
                            'outlier_zscore':['find outliers using zscore',
                                              'get zscore outliers',
                                              'z-score outliers',
                                              'get zscore outiers',
                                              'get z-score outliers'],
                              
                              
                              'outlier_norm': ['find outliers using normal distribution',
                                              'get outliers using normal distribution',
                                              'get outliers using norm-distribution',
                                              'get outliers using norm',
                                              'find outliers using normal distribution',
                                              'normal distribution outliers',
                                              'normal distribution outlier'],
                               
                               'outlier_dbscan' : ['find outliers using dbscan',
                                                  'find outlier usising dbscan',
                                                  'find outliers using DBSCAN',
                                                  'find outlier using DBSCAN',
                                                  'get outliers using dbscan',
                                                  'get outlier using dbscan'
                                                  'get outliers using DBSCAN',
                                                  'get outlier using DBSCAN']
                               
                              
                              })
                            
                            
info_outliers = {'outlier_iqr': {'module':'outliers',
                                'action':'action',
                                'topic':'topic',
                                'subtopic':'sub topic',
                                'input_format':'pd.DataFrame',
                                'description':'find outliers using inter quartile range (IQR)',
                                'arg_compat':'scale'},
              
             'outlier_zscore': {'module':'outliers',
                                'action':'action',
                                'topic':'topic',
                                'subtopic':'sub topic',
                                'input_format':'pd.DataFrame',
                                'description':'find outliers using zscore',
                                'arg_compat':'threshold'},
                 
                
             'outlier_norm': {'module':'outliers',
                                'action':'action',
                                'topic':'topic',
                                'subtopic':'sub topic',
                                'input_format':'pd.DataFrame',
                                'description':'find outliers using normal distribution',
                                'arg_compat':'threshold'},
                 
                 
             'outlier_dbscan': {'module':'outliers',
                                'action':'detect outliers',
                                'topic':'outlier detection',
                                'subtopic':'dbscan',
                                'input_format':'pd.DataFrame',
                                'description':"DBSCAN (Density-Based Spatial Clustering of Applications with Noise) can be used to detect outliers. DBSCAN is a clustering algorithm that groups together points that are close to each other based on a density criterion. Points that do not belong to any cluster are considered outliers or noise. The algorithm identifies these points by looking for areas of low density, where the distance between neighboring points is greater than a specified threshold. These points are then labeled as outliers. Therefore, DBSCAN can be used as an effective method for outlier detection in datasets where the majority of the data points form clusters.",
                                'arg_compat':'eps min_samples',},

                 
              }
                         
# configuration dictionary (passed in nlpm)
configure_outliers = {'corpus':corpus_outliers,'info':info_outliers}