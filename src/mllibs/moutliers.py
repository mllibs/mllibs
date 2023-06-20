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

class data_outliers(nlpi):
    
    # called in nlpm
    def __init__(self,nlp_config):
        self.name = 'outliers'          
        self.nlp_config = nlp_config  
        
    # called in nlpi
    def sel(self,args:dict):
                  
        select = args['pred_task']
            
        if(select == 'outlier_iqr'):
            self.outlier_iqr(args)
        elif(select == 'outlier_zscore'):
            self.outlier_zscore(args)
        elif(select == 'outlier_norm'):
            self.outlier_normal(args)
        elif(select == 'outlier_dbscan'):
            self.outlier_dbscan(args)

    @staticmethod
    def split_types(df):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']  
        numeric = df.select_dtypes(include=numerics)
        categorical = df.select_dtypes(exclude=numerics)
        return numeric,categorical
            
    # find outliers using IQR values
        
    @staticmethod
    def outlier_iqr(args:dict):
        
        df = args['data']

        # get the indicies of data outside 1.5 x IQR 
        
        if(args['scale'] is None):
            scale = 1.5  # set default
        else:
            scale = eval(args['scale']) # update value

        def get_iqroutlier(df):

            dict_outlier_index = {}
            dict_outlier = {}
            for k, v in df.items():
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
            for k,v in data.items():
                if(len(dict_outlier_index[v.name]) != 0):
                    ldata.loc[dict_outlier_index[v.name],'outlier_iqr'] = v.name
            return ldata
        
        ldata = label_outliers(df)
        nlpi.memory_output.append(ldata)

        
    # find outliers using z_scores

    def outlier_zscore(self,args:dict):
        
        df = args['data']
        num,cat = self.split_types(args['data'])
        
        if(args['threshold'] is None):
            threshold = 3
        else:
            threshold = eval(args['threshold'])
        
        def outliers_z_score(ys, threshold):
            mean_y = np.mean(ys)
            std_y = np.std(ys)
            z_scores = [(y - mean_y) / std_y for y in ys]
            return np.where(np.abs(z_scores) > threshold)[0]
    
        dict_outlier_index = {}
        for k, v in num.items():
            dict_outlier_index[v.name] = list(outliers_z_score(v,threshold))
            
        def label_outliers(data):
            ldata = data.copy()
            ldata.loc[:,'outlier_zscore'] = 'internal'
            for k,v in data.items():
                if(len(dict_outlier_index[v.name]) != 0):
                    ldata.loc[dict_outlier_index[v.name],'outlier_zscore'] = v.name
            return ldata
        
        ldata = label_outliers(df)
        nlpi.memory_output.append(ldata)
     
    # find outliers using normal distribution
    
    @staticmethod
    def outlier_normal(args:dict):
 
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
        
        if(args['threshold'] is None):
            threshold = 0.014
        else:
            threshold = eval(args['threshold'])
        
        df = args['data']
   
        dict_outlier_index = {}
        for k, v in df.items():
            
            # standardisation of columns
            w = v.to_frame()
            w_sc = StandardScaler().fit_transform(w)        
            v = pd.Series(w_sc[:,0],name=v.name)
            mu, sigma = estimate_gaussian(v.dropna())
            distribution = get_gaussian(mu, sigma)

            # calculate probability of the point appearing
            probabilities = get_probs(distribution,v.dropna())        
            dict_outlier_index[v.name] = np.where(probabilities < threshold)[0]
            
        def label_outliers(data):
            ldata = data.copy()
            ldata.loc[:,'outlier_normal'] = 'internal'
            for k,v in data.items():
                if(len(dict_outlier_index[v.name]) != 0):
                    ldata.loc[dict_outlier_index[v.name],'outlier_normal'] = v.name
            return ldata
        
        ldata = label_outliers(df)
        nlpi.memory_output.append(ldata)
        
    # find outliers using dbscan
        
    def outlier_dbscan(self,args:dict):
            
        df = args['data']
        num,cat = self.split_types(args['data'])
   
        w_sc = StandardScaler().fit_transform(num)        
        v = pd.DataFrame(w_sc,
                         columns = num.columns, 
                         index = num.index)
        
        if(args['eps'] is None):
            eps = 2.4
        else:
            eps = eval(args['eps'])
            
        if(args['min_samples'] is None):
            min_samples = 3
        else:
            min_samples = eval(args['min_samples'])
        
        db = DBSCAN(eps=eps,
                    min_samples=min_samples).fit(v)
        
        # for all features the same
        dict_outlier_index = OrderedDict({new_list: np.where(db.labels_ == -1)[0] for new_list in list(num.columns)})
        first = dict_outlier_index[[*dict_outlier_index.keys()][0]]

        def label_outliers(data):
            ldata = data.copy()
            ldata.loc[:,'outlier_dbscan'] = 'internal'
            ldata.loc[first,'outlier_dbscan'] = 'outlier'
            return ldata
            
        ldata = label_outliers(df)
        nlpi.memory_output.append(ldata)
        
            
    
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
                                'description':'find outliers using inter quartile range (IQR)'},
              
             'outlier_zscore': {'module':'outliers',
                                'action':'action',
                                'topic':'topic',
                                'subtopic':'sub topic',
                                'input_format':'pd.DataFrame',
                                'description':'find outliers using zscore'},
                 
                
             'outlier_norm': {'module':'outliers',
                                'action':'action',
                                'topic':'topic',
                                'subtopic':'sub topic',
                                'input_format':'pd.DataFrame',
                                'description':'find outliers using normal distribution'},
                 
                 
             'outlier_dbscan': {'module':'outliers',
                                'action':'action',
                                'topic':'topic',
                                'subtopic':'sub topic',
                                'input_format':'pd.DataFrame',
                                'description':'find outliers using unsupervised learning algorithm DBSCAN'},

                 
              }
                         
# configuration dictionary (passed in nlpm)
configure_outliers = {'corpus':corpus_outliers,'info':info_outliers}