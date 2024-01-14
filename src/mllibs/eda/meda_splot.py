from mllibs.dict_helper import sfp, sgp, sfpne, column_to_subset
from mllibs.nlpi import nlpi
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import warnings; warnings.filterwarnings('ignore')
from mllibs.nlpm import parse_json
from mllibs.df_helper import split_types
import pkg_resources
import json


# Define Palette
def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

palette = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252']
palette_rgb = [hex_to_rgb(x) for x in palette]

'''

Standard seaborn library visualisations

'''

class eda_splot(nlpi):
    
    def __init__(self):
        self.name = 'eda_splot'  

        path = pkg_resources.resource_filename('mllibs', '/eda/meda_splot.json')
        with open(path, 'r') as f:
            self.json_data = json.load(f)
            self.nlp_config = parse_json(self.json_data)
            
        #default_colors_p = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252'] # my custom (plotly)
        pallete = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
        self.default_colors = pallete
        
    # common functions
      
    def set_palette(self,args:dict):
      
      if(args['hue'] is not None):
          hueloc = args['data'][args['hue']]
          if(type(nlpi.pp['stheme']) is str):
              palette = nlpi.pp['stheme']
          else:
              palette = self.default_colors[:len(hueloc.value_counts())]
            
      else:
          hueloc = None
          palette = self.default_colors
        
      return palette

    def seaborn_setstyle(self):
      sns.set_style("whitegrid", {
          "ytick.major.size": 0.1,
          "ytick.minor.size": 0.05,
          'grid.linestyle': '--'
      })

    def sel(self,args:dict):
                
        select = args['pred_task']
        self.data_name = args['data_name']
        
        ''' 
        
        ADD EXTRA COLUMNS TO DATA 

        model_prediction | splits_col

        '''
        # split columns (tts,kfold,skfold) 
        if(len(nlpi.data[self.data_name]['splits_col']) != 0):

            split_dict = nlpi.data[self.data_name[0]]['splits_col']
            extra_columns = pd.concat(split_dict,axis=1)
            args['data'] = pd.concat([args['data'],extra_columns],axis=1)

        # model predictions
        if(len(nlpi.data[self.data_name]['model_prediction']) != 0):

            prediction_dict = nlpi.data[self.data_name[0]]['model_prediction']
            extra_columns = pd.concat(prediction_dict,axis=1)
            extra_columns.columns = extra_columns.columns.map('_'.join)
            args['data'] = pd.concat([args['data'],extra_columns],axis=1)

        ''' 
        
        Activatation Function
        
        '''

        if(select == 'sscatterplot'):
            self.sscatterplot(args)
        elif(select =='srelplot'):
            self.srelplot(args)
        elif(select == 'sboxplot'):
            self.sboxplot(args)
        elif(select == 'sresidplot'):
            self.sresidplot(args)
        elif(select == 'sviolinplot'):
            self.sviolinplot(args)
        elif(select == 'shistplot'):
            self.shistplot(args)
        elif(select == 'skdeplot'):
            self.skdeplot(args)
        elif(select == 'slmplot'):
            self.slmplot(args)
        elif(select == 'spairplot'):
            self.spairplot(args)
        elif(select == 'slineplot'):
            self.slineplot(args)
        elif(select == 'sheatmap'):
            self.sheatmap(args)
    
    '''
    
    Seaborn Scatter Plot [sns.scatterplot]
      
    '''
      
    def sscatterplot(self,args:dict):
          
        palette = self.set_palette(args)
        self.seaborn_setstyle()
        
        params = {
                  'data':args['data'],
                  'x':args['x'],
                  'y':args['y'],
                  'hue':args['hue'],
                  'alpha':args['alpha'],
                  'linewidth':args['mew'],
                  'edgecolor':args['mec'],
                  's':args['s'],
                  'palette':palette
                  }
          
        sns.scatterplot(**params)
        
        sns.despine(left=True, bottom=True)
        if(nlpi.pp['title']):
          plt.title(nlpi.pp['title'])
        plt.show()
        nlpi.resetpp()
        
    '''
    
    Seaborn scatter plot with Linear Model

      like relplot has [col] [row] options
      
    '''
        
    def slmplot(self,args:dict):
    
        self.seaborn_setstyle()
        
        sns.lmplot(x=args['x'], 
                   y=args['y'],
                   hue=args['hue'],
                   col=args['col'],
                   row=args['row'],
                   data=args['data']
                  )
        
        sns.despine(left=True, bottom=True)
        if(nlpi.pp['title']):
          plt.title(nlpi.pp['title'])
        plt.show()
        
    '''
    
    Seaborn Relation Plot

      main use to plot variation of scatterplot using [col] and [row] subsets
    
    '''

    def srelplot(self,args:dict):
            
        palette = self.set_palette(args)
        self.seaborn_setstyle()
        
        sns.relplot(x = args['x'], 
                    y = args['y'],
                    col = args['col'],
                    row = args['row'],
                    hue = args['hue'], 
                    col_wrap = args['col_wrap'],
                    palette = palette,
                    alpha = args['alpha'],
                    s = args['s'],
                    linewidth = args['mew'],
                    edgecolor = args['mec'],
                    data = args['data'])
        
        sns.despine(left=True, bottom=True)
        if(nlpi.pp['title']):
          plt.title(nlpi.pp['title'])
        plt.show()
        nlpi.resetpp()
        
    '''
    
    Seaborn Box Plot [sns.boxplot]
      
    '''
        
    def sboxplot(self,args:dict):
        
        palette = self.set_palette(args)
        self.seaborn_setstyle()
        
        if(args['bw'] is None):
            bw = 0.8
        else:
            bw = eval(args['bw'])
        
        sns.boxplot(x=args['x'], 
                    y=args['y'],
                    hue=args['hue'],
                    width=bw,
                    palette=palette,
                    data=args['data'])
        
        sns.despine(left=True, bottom=True)
        if(nlpi.pp['title']):
          plt.title(nlpi.pp['title'])
        plt.show()
        
    '''
    
    Seaborn Violin Plot [sns.violinplot]
      
    '''
        
    def sviolinplot(self,args:dict):
        
        palette = self.set_palette(args)
        self.seaborn_setstyle()
            
        sns.violinplot(x=args['x'], 
                       y=args['y'],
                       hue=args['hue'],
                       palette=palette,
                       data=args['data'],
                       inner="quart",
                       split=True
                       )   
        
        sns.despine(left=True, bottom=True)
        if(nlpi.pp['title']):
          plt.title(nlpi.pp['title'])
        plt.show()
        nlpi.resetpp()
        
    @staticmethod
    def sresidplot(args:dict):
      
        sns.residplot(x=args['x'], 
                      y=args['y'],
                      color=nlpi.pp['stheme'][1],
                      data=args['data'])
        
        sns.despine(left=True, bottom=True)
        plt.show()
        
    '''
    
    Seaborn Histogram Plot [sns.histplot]
      
    '''
      
    def shistplot(self,args:dict):
        
        self.seaborn_setstyle()
    
        # default parameters (pre) & allowable parameters (allow)
        pre = {'nbins':'auto','barmode':'stack'}
        allow = {'barmode':['layer','dodge','stack','fill']}
        
        # set default parameter if not set
        nbins = sfp(args,pre,'nbins')
        barmode = sfp(args,pre,'barmode')
        palette = self.set_palette(args)
        
        # check if string is in allowable parameter
        if(barmode not in allow['barmode']):
          barmode = allow['barmode'][0]
          print('[note] allowable barmodes: [layer],[dodge],[stack],[fill]')
          
        if(args['x'] is None and args['y'] is None and args['column'] is not None):
          args['x'] = args['column']
          print('[note] please specify orientation [x][y]')
        
        sns.histplot(
                      x=args['x'], 
                      y=args['y'],
                      hue=args['hue'],
                      alpha = args['alpha'],
                      linewidth=args['mew'],
                      edgecolor=args['mec'],
                      data=args['data'],
                      palette=palette,
                      bins=nbins,
                      multiple=barmode
        )
        
        sns.despine(left=True, bottom=True)
        if(nlpi.pp['title']): 
          plt.title(nlpi.pp['title'])
        plt.show()
        nlpi.resetpp()
        
    '''
    
    Seaborn Kernel Density Plot
    
    '''

    def skdeplot(self,args:dict):
          
        palette = self.set_palette(args)
            
        self.seaborn_setstyle()
        
        sns.kdeplot(x=args['x'],
                    y=args['y'],
                    hue = args['hue'],
                    palette=palette,
                    fill=nlpi.pp['fill'],
                    data = args['data'])
        
        sns.despine(left=True, bottom=True)
        if(nlpi.pp['title']):
          plt.title(nlpi.pp['title'])
        plt.show()
        nlpi.resetpp()
        
    def seaborn_pairplot(self,args:dict):
   
        num,cat = split_types(args['data'])
            
        if(args['hue'] is not None):
            hueloc = args['hue']
            num = pd.concat([num,args['data'][args['hue']]],axis=1) 
            subgroups = len(args['data'][args['hue']].value_counts())
            if(type(nlpi.pp['stheme']) is list):
                palette = nlpi.pp['stheme'][:subgroups]
            else:
                palette = nlpi.pp['stheme']
        else:
            hueloc = None
            palette = nlpi.pp['stheme']
        
            
        sns.set_style("whitegrid", {
            "ytick.major.size": 0.1,
            "ytick.minor.size": 0.05,
            'grid.linestyle': '--'
         })
             
        sns.pairplot(num,
                     hue=hueloc,
                     corner=True,
                     palette=palette,
                     diag_kws={'linewidth':nlpi.pp['mew'],
                               'fill':args['fill']},
                     plot_kws={'edgecolor':args['mec'],
                               'linewidth':args['mew'],
                               'alpha':args['alpha'],
                               's':args['s']})   
        
        sns.despine(left=True, bottom=True)
        plt.show()
        nlpi.resetpp()
        
    '''
    
    Seaborn Line Plot 
    
    '''

    def slineplot(self,args:dict):
    
        self.seaborn_setstyle()
        palette = self.set_palette(args)

        sns.lineplot(x=args['x'], 
                     y=args['y'],
                     hue=args['hue'],
                     alpha=args['alpha'],
                     linewidth=args['mew'],
                     data=args['data'],
                     palette=palette)
        
        sns.despine(left=True, bottom=True)
        if(nlpi.pp['title']):
          plt.title(nlpi.pp['title'])
        plt.show()
        nlpi.resetpp()

    # seaborn heatmap
                
    def sheatmap(self,args:dict):
        
        if(args['hue'] is not None):
            hueloc = args['data'][args['hue']]
            if(type(nlpi.pp['stheme']) is str):
                palette = nlpi.pp['stheme']
            else:
                palette = palette_rgb[:len(hueloc.value_counts())]
                
        else:
            hueloc = None
            palette = palette_rgb
        
        num,_ = self.split_types(args['data'])
        sns.heatmap(num,cmap=palette,
                    square=False,lw=2,
                    annot=True,cbar=True)    
                    
        plt.show()
        nlpi.resetpp()
    