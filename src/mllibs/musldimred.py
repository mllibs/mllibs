from collections import OrderedDict
from copy import deepcopy

from sklearn.decomposition import PCA,KernelPCA,IncrementalPCA,NMF,TruncatedSVD,FastICA
from sklearn.manifold import Isomap

from mllibs.nlpi import nlpi
from mllibs.nlpm import nlpm
import pandas as pd
import numpy as np
from mllibs.nlpm import parse_json
import json

'''

Dimensionality Reduction
decomposition/manifold

'''

class make_dimred(nlpi):
    
    def __init__(self):
        
        self.name = 'usldimred'  

        # read config data
        with open('src/mllibs/corpus/musldimred.json', 'r') as f:
            self.json_data = json.load(f)
            self.nlp_config = parse_json(self.json_data)

        self.select = None
        self.data = None
        self.args = None
        
    # verbose output
        
    @staticmethod
    def verbose_set(verbose):
        print(f'set {verbose}')
          
    # set function parameter (related to activation)
            
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
        
        
    # use only a subset of features
        
    def check_subset(self,args:dict):
        
        # if subset columns have been specified
        if(args['subset'] is not None):
            self.data = self.data[args['subset']]
            self.verbose_set('subset')
            
    # use only a sample of data
            
    def check_sample(self,args:dict):
        
        if(self.sgp(args,'sample') is not None):
            samples = self.sgp(args,'sample')
            print(samples)
            print(self.data.shape)
            self.data = self.data.sample(n=samples)
            print(f'sample data only: {self.data.shape[0]} samples')       
        
    # select activation function (called in NLPI)
    
    def sel(self,args:dict):
        
        self.select = args['pred_task']    
        self.data = deepcopy(args['data'])  # set main data       
        
        _,cat = nlpi.split_types(args['data']) # get categorical columns
        
        self.check_sample(args)  # sample row subset option (using self.data)        
        self.catn = self.data[cat]     
        self.check_subset(args)  # select feature subset option (using self.data)   
        self.args = args
            
        if(self.select == 'PCA'):
            self.pca(self.data,self.args)
        elif(self.select == 'kPCA'):
            self.kpca(self.data,self.args)
        elif(self.select == 'iPCA'):
            self.ipca(self.data,self.args)
        elif(self.select == 'NMF'):
            self.nmf(self.data,self.args)
        elif(self.select == 'tSVD'):
            self.tsvd(self.data,self.args)
        elif(self.select == 'fICA'):
            self.fica(self.data,self.args)
        elif(self.select == 'isomap'):
            self.isomap(self.data,self.args)

        
    ''' 
    
    Principal Component Analysis
    
    '''
        
    # Standard PCA
        
    def pca(self,data:pd.DataFrame,args:dict):
        
        # preset value dictionary
        pre = {'dim':2,'whiten':False}
        
        model = PCA(n_components=self.sfp(args,pre,'dim'),
                  whiten=self.sfp(args,pre,'whiten'))
        X = pd.DataFrame(model.fit_transform(data),index=self.catn.index)
        X.columns = [f'dim_{i}' for i in range(0,self.sfp(args,pre,'dim')) ]
    
        nlpi.memory_output.append({'data':pd.concat([X,self.catn],axis=1),
                                   'model':model})
        
    # Kernel PCA
        
    def kpca(self,data:pd.DataFrame,args:dict):

        # preset value dictionary
        pre = {'dim':2,'kernel':'linear'}
        data = deepcopy(data)
        
        model = KernelPCA(n_components=self.sfp(args,pre,'dim'),
                        kernel=self.sfp(args,pre,'kernel'))
        model.fit(data)
        X = pd.DataFrame(model.transform(data),index=self.catn.index)
        X.columns = [f'dim_{i}' for i in range(0,self.sfp(args,pre,'dim')) ]
    
        nlpi.memory_output.append({'data':pd.concat([X,self.catn],axis=1),
                                   'model':model}) 
        
    # Iterative PCA
        
    def ipca(self,data:pd.DataFrame,args:dict):
        
        # preset value dictionary
        pre = {'dim':2,'batch':10}
        data = deepcopy(data)
        
        model = IncrementalPCA(n_components=self.sfp(args,pre,'dim'),
                             batch_size= self.sfp(args,pre,'batch'))
        model.fit(data)
        X = pd.DataFrame(model.transform(data),index=self.catn.index)
        X.columns = [f'dim_{i}' for i in range(0,self.sfp(args,pre,'dim')) ]
    
        nlpi.memory_output.append({'data':pd.concat([X,self.catn],axis=1),
                                   'model':model}) 
        
    '''
    
    Non-Negative Matrix Factorisation
    
    '''

    def nmf(self,data:pd.DataFrame,args:dict):
        
        # preset value dictionary
        pre = {'dim':2}
        data = deepcopy(data)
        
        model = NMF(n_components=self.sfp(args,pre,'dim'))
        model.fit(data)
        X = pd.DataFrame(model.transform(data),index=self.catn.index)
        X.columns = [f'dim_{i}' for i in range(0,self.sfp(args,pre,'dim')) ]
    
        nlpi.memory_output.append({'data':pd.concat([X,self.catn],axis=1),
                                   'model':model})         

    # Tuncated SVD 
        
    def tsvd(self,data:pd.DataFrame,args:dict):
        
        # preset value dictionary
        pre = {'dim':2}
        data = deepcopy(data)
        
        model = TruncatedSVD(n_components=self.sfp(args,pre,'dim'))
        model.fit(data)
        X = pd.DataFrame(model.transform(data),index=self.catn.index)
        X.columns = [f'dim_{i}' for i in range(0,self.sfp(args,pre,'dim')) ]
    
        nlpi.memory_output.append({'data':pd.concat([X,self.catn],axis=1),
                                   'model':model}) 
                        
        
    # Fast ICA decomposition
        
    def fica(self,data:pd.DataFrame,args:dict):
        
        # preset value dictionary
        pre = {'dim':2,'whiten_solver':'svd','whiten':'arbitrary-variance'}
        data = deepcopy(data)
        
        model = FastICA(n_components=self.sfp(args,pre,'dim'),
                        whiten=self.sfp(args,pre,'whiten'),
                        whiten_solver=self.sfp(args,pre,'whiten_solver'))
        model.fit(data)
        X = pd.DataFrame(model.transform(data),index=self.catn.index)
        X.columns = [f'dim_{i}' for i in range(0,self.sfp(args,pre,'dim')) ]
    
        nlpi.memory_output.append({'data':pd.concat([X,self.catn],axis=1),
                                   'model':model}) 
        
    # Isomap Manifold 
        
    def isomap(self,data:pd.DataFrame,args:dict):
        
        # preset value dictionary
        pre = {'dim':2,'n_neighbours':5,'radius':None}
        data = deepcopy(data)
        
        model = Isomap(n_components=self.sfp(args,pre,'dim'),
                        n_neighbors=self.sfp(args,pre,'n_neighbours'),
                        radius=self.sfp(args,pre,'radius'))
        model.fit(data)
        X = pd.DataFrame(model.transform(data),index=self.catn.index)
        X.columns = [f'dim_{i}' for i in range(0,self.sfp(args,pre,'dim')) ]
    
        nlpi.memory_output.append({'data':pd.concat([X,self.catn],axis=1),
                                   'model':model}) 
