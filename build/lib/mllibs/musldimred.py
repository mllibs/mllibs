from collections import OrderedDict
from copy import deepcopy

from sklearn.decomposition import PCA,KernelPCA,IncrementalPCA,NMF,TruncatedSVD,FastICA
from sklearn.manifold import Isomap

from mllibs.nlpi import nlpi
from mllibs.nlpm import nlpm
import pandas as pd
import numpy as np


'''

Dimensionality Reduction
decomposition/manifold

'''

# sample module class structure
class usldimred(nlpi):
    
    # called in nlpm
    def __init__(self,nlp_config):
        
        # unique module name identifier (used in nlpm/nlpi)
        self.name = 'usldimred'  
        
        # text based info related to module (used in nlpm/nlpi)
        self.nlp_config = nlp_config  
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
    
'''

Corpus

'''

corpus_usldimred = OrderedDict({"PCA":['PCA dimension reduction',
                                       'PCA reduce dimension',
                                       'PCA dimred',
                                       'PCA dimensionality reduction',
                                       'principal component analysis',
                                       'principal component analysis dimension reduction',
                                       'dimred PCA',
                                       'PCA decomposition',
                                       'lower dimensionality PCA'],
                            
                                'kPCA':['kernel PCA dimensionality reduction',
                                        'kPCA dimension reduction',
                                        'kPCA dimensionality reduction',
                                        'reducing dimension kPCA',
                                        'kPCA dimred',
                                        'kPCA reduce dimension',
                                        'kPCA reduce dimensions', 
                                    ],
                                
                                 'iPCA':['incremental PCA dimensionality reduction',
                                         'incremental PCA',
                                         'incremental PCA dimension reduction',
                                         'incremental principal component analysis',
                                         'increment PCA',
                                         'iPCA dimensionality reduction',
                                         'batch PCA'],
                                
                                
                                'NMF' : ['non negative matrix factorisation',
                                        'NMF factorisation',
                                        'NMF matrix factorisation',
                                        'NMF decomposition',
                                        'NMF dimensionality reduction',
                                        'non-negative matrix factorisation'],
                                
                                'tSVD' : ['tuncated SVD',
                                         'tuncated SVD matrix factorisation',
                                         'truncated SVD decomposition',
                                         'tSVD decomposition',
                                         'tSVD matrix factorisation',
                                          'dimensionality reduction using truncated SVD',
                                          'dimension reduction SVD',
                                          'dimension reduction truncated SVD'
                                         ],
                                
                                'fICA' : ['fast ICA',
                                         'fast ICA decomposition',
                                         'fast independent component analysis',
                                         'fast ICA dimension reduction',
                                         'fICA decomposition',
                                          'fICA factorisation',
                                         'dimensionality reduction fast ICA'],
                                
                                'isomap' : ['isomap embedding',
                                            'isometric mapping',
                                            'isomap dimension reduction',
                                            'isometric map',
                                            'isomap embedding manifold learning',
                                            'isomap embedding manifold',
                                            'manifold learning isomap',
                                            'isomap manifold learning dimension reduction']
                                
                                
                               })

info_usldimred = {'PCA': {'module':'usldimred',
                            'action':'reduce dimensions',
                            'topic':'dimensionality reduction',
                            'subtopic':'decomposition',
                            'input_format':'pd.DataFrame',
                            'description':'Principal component analysis (PCA) decomposition, project data to lower dimensional space. Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. Input data is centered but not scaled for each feature before applying the SVD',
                            'token_compat':'data subset sample',
                            'arg_compat':'dim whiten'},
                  
                  
              'kPCA':{'module':'usldimred',
                            'action':'reduce dimensions',
                            'topic':'dimensionality reduction',
                            'subtopic':'decomposition',
                            'input_format':'pd.DataFrame',
                            'description':'Kernel principal component analysis (KPCA) is a non-linear extension of the traditional PCA algorithm. It involves applying a non-linear transformation to the data before performing PCA, which allows for more complex patterns and relationships to be captured. This transformation is achieved using a kernel function, which maps the original data into a higher-dimensional space where it may be more separable. The KPCA algorithm then performs PCA on this transformed data to identify the most important components or features',
                            'token_compat':'data subset sample',
                            'arg_compat':'dim kernel'},

                  
              'iPCA':{'module':'usldimred',
                      'action':'reduce dimensions',
                      'topic':'dimensionality reduction',
                      'subtopic':'decomposition',
                      'input_format':'pd.DataFrame',
                      'description':'Incremental principal components analysis (IPCA) is a variant of the traditional PCA algorithm that allows for the incremental computation of principal components as new data becomes available. In traditional PCA, all of the data must be available at once in order to perform the computation. IPCA, on the other hand, allows for the addition of new data points one at a time, which can be useful in situations where data is arriving continuously or in batches. IPCA works by updating the covariance matrix and eigenvalues of the data incrementally, rather than computing them from scratch each time new data is added. This can save computational time and resources, especially when dealing with large datasets',
                      'token_compat':'data subset sample',
                      'arg_compat':'dim kernel batch'
                      },


              'NMF':{'module':'usldimred',
                      'action':'reduce dimensions',
                      'topic':'dimensionality reduction',
                      'subtopic':'decomposition',
                      'input_format':'pd.DataFrame',
                      'description':'Non-negative matrix factorization (NMF) is a technique that decomposes a non-negative matrix into two lower-rank non-negative matrices. The goal of NMF is to find a low-dimensional representation of the original data that captures the underlying patterns and structure of the data. NMF is often used for feature extraction, dimensionality reduction, and clustering in a variety of applications such as image processing, text mining, and bioinformatics. In NMF, the original matrix is factorized into two matrices: a basis matrix and a coefficient matrix. The basis matrix represents the basis vectors that form the low-dimensional representation of the data, while the coefficient matrix represents the weights of these basis vectors for each data point. The basis matrix and coefficient matrix are typically learned through an iterative optimization process that minimizes the reconstruction error between the original data and the low-dimensional representation. NMF has several advantages over other dimensionality reduction techniques such as PCA, including the ability to handle non-negative data and the ability to extract interpretable features. However, NMF can be sensitive to initialization and may not always converge to a global minimum',
                      'token_compat':'data subset sample',
                      'arg_compat':'dim'},
                  
              'tSVD':{'module':'usldimred',
                      'action':'reduce dimensions',
                      'topic':'dimensionality reduction',
                      'subtopic':'decomposition',
                      'input_format':'pd.DataFrame',
                      'description':'Truncated SVD (Singular Value Decomposition) is a matrix factorization technique that decomposes a matrix into three matrices: a left singular matrix, a diagonal singular value matrix, and a right singular matrix. Truncated SVD is similar to NMF in that it is used for dimensionality reduction and feature extraction, but it can handle both positive and negative data. The "truncated" part of the name refers to the fact that only a subset of the singular values and corresponding singular vectors are retained, while the rest are discarded. This results in a lower-dimensional approximation of the original matrix that captures the most important information',
                      'token_compat':'data subset sample',
                      'arg_compat':'dim'},
                  
              'fICA':{'module':'usldimred',
                      'action':'reduce dimensions',
                      'topic':'dimensionality reduction',
                      'subtopic':'decomposition',
                      'input_format':'pd.DataFrame',
                      'description':'Independent component analysis separates a multivariate signal into additive subcomponents that are maximally independent. It is implemented in scikit-learn using the Fast ICA algorithm. Typically, ICA is not used for reducing dimensionality but for separating superimposed signals. Since the ICA model does not include a noise term, for the model to be correct, whitening must be applied. This can be done internally using the whiten argument or manually using one of the PCA variants',
                      'token_compat':'data subset sample',
                      'arg_compat':'dim whiten whiten_solver'},
                  
                  
              'isomap':{'module':'usldimred',
                      'action':'reduce dimensions',
                      'topic':'dimensionality reduction',
                      'subtopic':'manifold learning',
                      'input_format':'pd.DataFrame',
                      'description':"Isomap embedding is a nonlinear dimensionality reduction technique that preserves the intrinsic geometric structure of high-dimensional data in a lower-dimensional space. It is based on the concept of geodesic distances, which measure the shortest path between two points on a manifold (a curved space). Isomap works by first constructing a graph of nearest neighbors for the data points, and then computing the shortest path distances between all pairs of points on this graph using a variant of Dijkstra's algorithm. These distances are then used to construct a low-dimensional embedding of the data using classical multidimensional scaling (MDS)",
                      'token_compat':'data subset sample',
                      'arg_compat':'dim n_neighbors radius'},
                        
                 }
                  
# configuration dictionary (passed in nlpm)
configure_usldimred = {'corpus':corpus_usldimred,'info':info_usldimred}