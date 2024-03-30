
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

#       default_colors_p = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252'] # my custom (plotly)


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
            
        pallete = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
        self.default_colors = pallete

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
            
    # subset treatment 
    @staticmethod
    def check_column_format(lst):
      
      if(len(lst) == 1):
        return lst[0]
      elif(len(lst) == 2):
        print("[note] I'll group the specified columns together")
        
        # nested lists to single list
        # taking into account str entries
        grouped = []
        for item in lst:
          if isinstance(item, list):
            grouped.extend(item)
          elif isinstance(item, str):
            grouped.append(item)
          else:
            grouped.append(item)
            
        return grouped 
      
      else:
        print('[note] please specify the columns you want to transform only')
        return None

    '''
  
    For each specified column plot the univariate KDE plot

      [column] select the columns for which kde to plot in plt.subplots
               if columns are not specified, all numerical columns are displayed

    '''
        
    def eda_colplot_kde(self,args:dict):
      
        sns.set_style("whitegrid", {'grid.linestyle': '--'})
      
        # only specified columns or all columns
        if(args['column'] != None):
          
            # single subset group only
            columns = self.check_column_format(args['column'])
            
        else:
            columns,_ = split_types(args['data']) # numeric column names only

        n_cols = 3
        n_rows = math.ceil(len(columns)/n_cols)

        if(nlpi.pp['figsize'] == None or nlpi.pp['figsize'][0] > 20):
          fsize = (16, n_rows*5)
        else:
          fsize = nlpi.pp['figsize']
        
        fig, ax = plt.subplots(n_rows, n_cols, figsize=fsize)
        ax = ax.flatten()
      
        for i, column in enumerate(columns):
            plot_axes = [ax[i]]
          
            sns.kdeplot(data=args['data'],
                        x=column,
                        fill=True,
                        hue=args['hue'],
                        # fill=nlpi.pp['fill'],
                        # alpha= nlpi.pp['alpha'],
                        # linewidth=nlpi.pp['mew'],
                        # edgecolor=nlpi.pp['mec'],
                        ax=ax[i],
                        legend=nlpi.pp['legend'],
                        common_norm=False,
                        )
          
            sns.despine(top=True,left=True)
          
        plt.tight_layout()
        plt.show()
        
    '''
    
    For each specified column plot a univariate boxplot 

    [column] select the columns for which boxplot to plot in plt.subplots
              if columns are not specified, all numerical columns are displayed

         [x] x based variation for different subgroups (categorical only)
      
    '''

    def eda_colplot_box(self,args:dict):
      
      sns.set_style("whitegrid", {'grid.linestyle': '--'})
      
      # only specified columns or all columns
      if(args['column'] != None):
        
          # single subset group only
          columns = self.check_column_format(args['column'])
        
      else:
          columns,_ = split_types(args['data']) # numeric column names only
        
      n_cols = 3
      n_rows = math.ceil(len(columns)/n_cols)
      
      if(nlpi.pp['figsize'] == None or nlpi.pp['figsize'][0] > 20):
        fsize = (16, n_rows*5)
      else:
        fsize = nlpi.pp['figsize']
        
      fig, ax = plt.subplots(n_rows, n_cols, figsize=fsize)
      sns.despine(fig, left=True, bottom=True)
      ax = ax.flatten()
      
      for i, column in enumerate(columns):
        plot_axes = [ax[i]]
        
        sns.boxplot(
          args['data'],        # dataset
          y=column,            # looping column name
          x=args['x'],         # x wide variation for category
          width=nlpi.pp['bw'], # boxwidth 
          ax=ax[i])
        
        sns.despine(top=True,left=True)
        
      plt.tight_layout()
      plt.show()

    # column scatter plot for numeric columns only
    
    '''
    
    For each specified column plot a scatterplot 
    
    [column] select the columns for which boxplot to plot in plt.subplots
              if columns are not specified, all numerical columns are displayed
    
          [x] numeric or categorical column
      
    '''
        
    def eda_colplot_scatter(self,args:dict):

        sns.set_style("whitegrid", {'grid.linestyle': '--'})
        sns.set_palette(self.default_colors)

        # only specified columns or all columns
        if(args['column'] != None):
          
            # single subset group only
            columns = self.check_column_format(args['column'])
          
        else:
            columns,_ = split_types(args['data']) # numeric column names only
            
        # remove itself if args['x'] used
        if(args['x'] in columns):
          columns.remove(args['x'])
          
        n_cols = 3
        n_rows = math.ceil(len(columns)/n_cols)
            
        if(nlpi.pp['figsize'] == None or nlpi.pp['figsize'][0] > 20):
          fsize = (16, n_rows*5)
        else:
          fsize = nlpi.pp['figsize']
        
        fig, ax = plt.subplots(n_rows, n_cols, figsize=fsize)
        ax = ax.flatten()

        for i, column in enumerate(columns):
          
            plot_axes = [ax[i]]

            sns.scatterplot(
                args['data'], 
                y=column,
                x=args['x'],
                hue=args['hue'],
                alpha= args['alpha'],
                linewidth=args['mew'],
                edgecolor=args['mec'],
                s = args['s'],
                ax=ax[i],
            )
          
            sns.despine(top=True,left=True)
        
        plt.tight_layout()
        plt.show()