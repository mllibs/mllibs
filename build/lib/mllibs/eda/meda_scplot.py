
from mllibs.nlpi import nlpi
from mllibs.df_helper import split_types
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math
from collections import OrderedDict
import warnings; warnings.filterwarnings('ignore')
from mllibs.nlpm import parse_json
import pkg_resources
import json


def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

palette = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252']
palette_rgb = [hex_to_rgb(x) for x in palette]


'''

Feature Column based visualisations using seaborn

'''

class eda_scplot(nlpi):
    
    def __init__(self):
        self.name = 'eda_scplot'      

        path = pkg_resources.resource_filename('mllibs', '/eda/meda_scplot.json')
        with open(path, 'r') as f:
            self.json_data = json.load(f)
            self.nlp_config = parse_json(self.json_data)

    '''

    Select Activation Function

    '''
        
    def sel(self,args:dict):
        
        select = args['pred_task']
                  
        if(select == 'col_kde'):
            self.eda_colplot_kde(args)
        elif(select == 'col_box'):
            self.eda_colplot_box(args)
        elif(select == 'col_scatter'):
            self.eda_colplot_scatter(args)

    '''
    
    Activation Functions
    
    '''

    # column KDE plots for numeric columns
        
    def eda_colplot_kde(self,args:dict):
        
        # get numeric column names only
        num,_ = split_types(args['data'])
            
        if(args['x'] is not None):
            xloc = args['data'][args['x']]
        else:
            xloc = None
          
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
                         )
    
            # titles
            ax[i].set_title(f'{column} distribution');
            ax[i].set_xlabel(None)
    
        for i in range(i+1, len(ax)):
            ax[i].axis('off')
                      
        plt.tight_layout()
        
    # column boxplots for numeric columns

    def eda_colplot_box(self,args:dict):

        # split data into numeric & non numeric
        num,cat = split_types(args['data'])
          
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
            )

            # titles
            ax[i].set_title(f'{column} distribution');
            ax[i].set_xlabel(None)
            
            
        for i in range(i+1, len(ax)):
            ax[i].axis('off')
        
        plt.tight_layout()

    # column scatter plot for numeric columns only
        
    def eda_colplot_scatter(self,args:dict):

        # split data into numeric & non numeric
        num,_ = split_types(args['data'])
          
        columns = list(num.columns)  
        n_cols = 3
        n_rows = math.ceil(len(columns)/n_cols)
        
        if(args['x'] is not None):
            xloc = args['data'][args['x']]
        else:
            xloc = None
        
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
                alpha= nlpi.pp['alpha'],
                linewidth=nlpi.pp['mew'],
                edgecolor=nlpi.pp['mec'],
                s = nlpi.pp['s'],
                ax=ax[i],
            )

            # titles
            ax[i].set_title(f'{column} distribution');
            ax[i].set_xlabel(None)
            
            
        for i in range(i+1, len(ax)):
            ax[i].axis('off')
        
        plt.tight_layout()
        plt.show()