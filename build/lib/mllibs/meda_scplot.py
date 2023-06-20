
from mllibs.nlpi import nlpi
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math
from collections import OrderedDict
import warnings; warnings.filterwarnings('ignore')


def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

palette = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252']
palette_rgb = [hex_to_rgb(x) for x in palette]


'''



Feature Column based visualisations using seaborn



'''

class eda_colplot(nlpi):
    
    # called in nlpm
    def __init__(self,nlp_config):
        self.name = 'eda_colplot'          
        self.nlp_config = nlp_config  
        
    # called in nlpi
    def sel(self,args:dict):
        
        select = args['pred_task']
                  
        if(select == 'col_kde'):
            self.eda_colplot_kde(args)
        elif(select == 'col_box'):
            self.eda_colplot_box(args)
        elif(select == 'col_scatter'):
            self.eda_colplot_scatter(args)
            
    @staticmethod
    def split_types(df):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']  
        numeric = df.select_dtypes(include=numerics)
        categorical = df.select_dtypes(exclude=numerics)
        return numeric,categorical
        
    def eda_colplot_kde(self,args:dict):
        
        num,cat = self.split_types(args['data'])
            
        if(args['x'] is not None):
            xloc = args['data'][args['x']]
        else:
            xloc = None
            
        if(args['hue'] is not None):
            hueloc = args['data'][args['hue']]
            if(type(nlpi.pp['stheme']) is str):
                palette = nlpi.pp['stheme']
            else:
                palette = palette_rgb[:len(hueloc.value_counts())]
                
        else:
            hueloc = None
            palette = palette_rgb
          
        columns = list(num.columns)  
        n_cols = 3
        n_rows = math.ceil(len(columns)/n_cols)

        fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, n_rows*5))
        ax = ax.flatten()

        for i, column in enumerate(columns):
            plot_axes = [ax[i]]
            
            sns.set_style("whitegrid", {
            'grid.linestyle': '--'})
    
            sns.kdeplot(data=args['data'],
                        x=column,
                        hue=hueloc,
                        fill=nlpi.pp['fill'],
                        alpha= nlpi.pp['alpha'],
                        linewidth=nlpi.pp['mew'],
                        edgecolor=nlpi.pp['mec'],
                        ax=ax[i],
                        common_norm=False,
                        palette=palette
                         )
    
            # titles
            ax[i].set_title(f'{column} distribution');
            ax[i].set_xlabel(None)
    
        for i in range(i+1, len(ax)):
            ax[i].axis('off')
                      
        plt.tight_layout()
        
    def eda_colplot_box(self,args:dict):

        # split data into numeric & non numeric
        num,cat = self.split_types(args['data'])
          
        columns = list(num.columns)  
        n_cols = 3
        n_rows = math.ceil(len(columns)/n_cols)
        
        if(args['x'] is not None):
            xloc = args['data'][args['x']]
        else:
            xloc = None
            
        if(args['x'] is not None):
            xloc = args['data'][args['x']]
        else:
            xloc = None
            
        if(args['hue'] is not None):
            hueloc = args['data'][args['hue']]
            if(type(nlpi.pp['stheme']) is str):
                palette = nlpi.pp['stheme']
            else:
                palette = palette_rgb[:len(hueloc.value_counts())]
                
        else:
            hueloc = None
            palette = palette_rgb

        fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, n_rows*5))
        sns.despine(fig, left=True, bottom=True)
        ax = ax.flatten()

        for i, column in enumerate(columns):
            plot_axes = [ax[i]]
            
            sns.set_style("whitegrid", {
            'grid.linestyle': '--'})


            if(args['bw'] is None):
                bw = 0.8
            else:
                bw = eval(args['bw'])

            sns.boxplot(
                y=args['data'][column],
                x=xloc,
                hue=hueloc,
                width=bw,
                ax=ax[i],
                palette=palette
            )

            # titles
            ax[i].set_title(f'{column} distribution');
            ax[i].set_xlabel(None)
            
            
        for i in range(i+1, len(ax)):
            ax[i].axis('off')
        
        plt.tight_layout()
        
    def eda_colplot_scatter(self,args:dict):

        # split data into numeric & non numeric
        num,cat = self.split_types(args['data'])
          
        columns = list(num.columns)  
        n_cols = 3
        n_rows = math.ceil(len(columns)/n_cols)
        
        if(args['x'] is not None):
            xloc = args['data'][args['x']]
        else:
            xloc = None
            
        if(args['hue'] is not None):
            hueloc = args['data'][args['hue']]
            if(type(nlpi.pp['stheme']) is str):
                palette = nlpi.pp['stheme']
            else:
                palette = palette_rgb[:len(hueloc.value_counts())]
                
        else:
            hueloc = None
            palette = palette_rgb

        fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, n_rows*5))
        sns.despine(fig, left=True, bottom=True)
        ax = ax.flatten()

        for i, column in enumerate(columns):
            plot_axes = [ax[i]]
            
            sns.set_style("whitegrid", {
            'grid.linestyle': '--'})

            sns.scatterplot(
                y=args['data'][column],
                x=xloc,
                hue=hueloc,
                alpha= nlpi.pp['alpha'],
                linewidth=nlpi.pp['mew'],
                edgecolor=nlpi.pp['mec'],
                s = nlpi.pp['s'],
                ax=ax[i],
                palette=palette,
            )

            # titles
            ax[i].set_title(f'{column} distribution');
            ax[i].set_xlabel(None)
            
            
        for i in range(i+1, len(ax)):
            ax[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    

corpus_colplot = OrderedDict({"col_kde":['plot column feature kernel density estimation',
                                       'plot column kernel density estimation',
                                       'plot column kde',
                                        'visualise column feature kernel density estimation',
                                        'visualise column kernel density estimation',
                                        'visualise column kde'
                                       ],
                             
                             
                             'col_box':['plot column feature boxplot',
                                       'plot column feature box plot',
                                       'plot column boxplot',
                                       'plot column box plot',
                                       'visualise column feature boxplot',
                                       'visualise column boxplot',
                                       'visualise column box plot',
                                       ],
                              
                             'col_scatter':['plot column feature scatterplot',
                                       'plot column feature scatter plot',
                                       'plot column scatterplot',
                                       'plot column scatter plot',
                                       'visualise column feature scatterplot',
                                       'visualise column scatterplot',
                                       'visualise column scatter plot',
                                       ],

                             
                             })
                            
info_colplot = {'col_kde':    {'module':'eda_colplot',
                              'action':'show plot',
                              'topic':'exploratory data analysis (eda)',
                              'subtopic':'column eda',
                              'input_format':'pd.DataFrame',
                              'description':'visualise/plot column feature kernel density estimation plot'},
               
               'col_box':     {'module':'eda_colplot',
                              'action':'show plot',
                              'topic':'exploratory data analysis (eda)',
                              'subtopic':'column eda',
                              'input_format':'pd.DataFrame',
                              'description':'visualise/plot column feature boxplot'},
                

               'col_scatter':     {'module':'eda_colplot',
                                  'action':'show plot',
                                  'topic':'exploratory data analysis (eda)',
                                  'subtopic':'column eda',
                                  'input_format':'pd.DataFrame',
                                  'description':'visualise/plot column feature scatterplot'}
               
               }
                         
# configuration dictionary (passed in nlpm)
configure_colplot = {'corpus':corpus_colplot,'info':info_colplot}