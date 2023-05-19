from scipy.stats import norm
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

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
#         elif(select == 'outlier_dbscan'):
            
    # find outliers using IQR values
        
    @staticmethod
    def outlier_iqr(args:dict):

        # get the indicies of data outside 1.5 x IQR 

        def get_iqroutlier(df):

            dict_outlier_index = {}
            dict_outlier = {}
            for k, v in df.items():
                q1 = v.quantile(0.25);
                q3 = v.quantile(0.75);
                irq = q3 - q1

                # select data
                v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
                dict_outlier_index[v_col.name] = list(v_col.index)

            return dict_outlier_index
        
        # return dictionary containing indicies of outliers
        dict_index = get_iqroutlier(args['data'])
        nlpi.memory_output.append(dict_index)
        
    # find outliers using z_scores

    @staticmethod
    def outlier_zscore(args:dict):

        def outliers_z_score(ys, threshold=3):
            mean_y = np.mean(ys)
            std_y = np.std(ys)
            z_scores = [(y - mean_y) / std_y for y in ys]
            return np.where(np.abs(z_scores) > threshold)[0]
       
        df = args['data']
   
        dict_outlier_index = {}
        for k, v in df.items():
            dict_outlier_index[v.name] = outliers_z_score(v)
            
        nlpi.memory_output.append(dict_outlier_index)
     
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
        
        df = args['data']
   
        dict_outlier_index = {}
        for k, v in df.items():
            
            # create normal distribution
            mu, sigma = estimate_gaussian(v.dropna())
            distribution = get_gaussian(mu, sigma)

            # calculate probability of the point appearing
            probabilities = get_probs(distribution,v.dropna())
            
            dict_outlier_index[v.name] = np.where(probabilities > 95)[0]
        
        nlpi.memory_output.append(dict_outlier_index)
        
        
        
            
    
corpus_outliers = OrderedDict({"outlier_iqr":['find outliers in data using IQR',
                                           'find outliers using IQR',
                                           'get IQR outliers',
                                           'find IQR outliers',
                                           'find IQR outlier',
                                           'inter quartile range outliers',
                                           'boxplot outliers'],
                            
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
                               
                              
                              })
                            
                            
info_outliers = {'outlier_iqr': {'module':'outliers',
                                'action':'action',
                                'topic':'topic',
                                'subtopic':'sub topic',
                                'input_format':'pd.DataFrame',
                                'output_format':'dict',
                                'description':'find outliers using inter quartile range (IQR)'},
              
             'outlier_zscore': {'module':'outliers',
                                'action':'action',
                                'topic':'topic',
                                'subtopic':'sub topic',
                                'input_format':'pd.DataFrame',
                                'output_format':'dict',
                                'description':'find outliers using zscore'},
                 
                
             'outlier_norm': {'module':'outliers',
                                'action':'action',
                                'topic':'topic',
                                'subtopic':'sub topic',
                                'input_format':'pd.DataFrame',
                                'output_format':'dict',
                                'description':'find outliers using norma distribution'},

                 
              }
                         
# configuration dictionary (passed in nlpm)
configure_outliers = {'corpus':corpus_outliers,'info':info_outliers}