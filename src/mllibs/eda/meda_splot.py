
from mllibs.nlpi import nlpi
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import warnings; warnings.filterwarnings('ignore')
from mllibs.nlpm import parse_json
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
        
    # called in nlpi

    def sel(self,args:dict):
                
        select = args['pred_task']
        self.data_name = args['data_name']
        
        ''' 
        
        ADD EXTRA COLUMNS TO DATA 

        model_prediction | splits_col

        
        '''
        # split columns (tts,kfold,skfold) 
        if(len(nlpi.data[self.data_name[0]]['splits_col']) != 0):

            split_dict = nlpi.data[self.data_name[0]]['splits_col']
            extra_columns = pd.concat(split_dict,axis=1)
            args['data'] = pd.concat([args['data'],extra_columns],axis=1)

        # model predictions
        if(len(nlpi.data[self.data_name[0]]['model_prediction']) != 0):

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
            self.seaborn_relplot(args)
        elif(select == 'sboxplot'):
            self.seaborn_boxplot(args)
        elif(select == 'sresidplot'):
            self.sresidplot(args)
        elif(select == 'sviolinplot'):
            self.sviolinplot(args)
        elif(select == 'shistplot'):
            self.shistplot(args)
        elif(select == 'skdeplot'):
            self.seaborn_kdeplot(args)
        elif(select == 'slmplot'):
            self.slmplot(args)
        elif(select == 'spairplot'):
            self.spairplot(args)
        elif(select == 'slineplot'):
            self.slineplot(args)
        elif(select == 'sheatmap'):
            self.sheatmap(args)

    # Seaborn Scatter Plot
      
    @staticmethod
    def sscatterplot(args:dict):
            
        sns.set_style("whitegrid", {
            "ytick.major.size": 0.1,
            "ytick.minor.size": 0.05,
            'grid.linestyle': '--'
        })

        sns.scatterplot(x=args['x'], 
                        y=args['y'],
                        hue=args['hue'],
                        alpha = nlpi.pp['alpha'],
                        linewidth=nlpi.pp['mew'],
                        edgecolor=nlpi.pp['mec'],
                        s = nlpi.pp['s'],
                        data=args['data'])
        
        sns.despine(left=True, bottom=True)
        plt.show()
        nlpi.resetpp()
        

    @staticmethod
    def seaborn_lmplot(args:dict):
    
        sns.set_style("whitegrid", {
            "ytick.major.size": 0.1,
            "ytick.minor.size": 0.05,
            'grid.linestyle': '--'})
        
        sns.lmplot(x=args['x'], 
                   y=args['y'],
                   hue=args['hue'],
                   col=args['col'],
                   row=args['row'],
                   data=args['data'])
        
        sns.despine(left=True, bottom=True)
        plt.show()

    @staticmethod
    def seaborn_relplot(args:dict):
            
        if(args['hue'] is not None):
            hueloc = args['data'][args['hue']]
            if(type(nlpi.pp['stheme']) is str):
                palette = nlpi.pp['stheme']
            else:
                palette = palette_rgb[:len(hueloc.value_counts())]
                
        else:
            hueloc = None
            palette = palette_rgb           
            
        sns.set_style("whitegrid", {
            "ytick.major.size": 0.1,
            "ytick.minor.size": 0.05,
            'grid.linestyle': '--'
        })
        
        sns.relplot(x=args['x'], 
                    y=args['y'],
                    col=args['col'],
                    row=args['row'],
                    hue=args['hue'], 
                    col_wrap=args['col_wrap'],
                    kind=args['kind'],
                    palette=palette,
                    alpha= nlpi.pp['alpha'],
                    s = nlpi.pp['s'],
                    linewidth=nlpi.pp['mew'],
                    edgecolor=nlpi.pp['mec'],
                    data=args['data'])
        
        sns.despine(left=True, bottom=True)
        plt.show()
        nlpi.resetpp()
        
    @staticmethod
    def seaborn_boxplot(args:dict):
        
        if(args['hue'] is not None):
            hueloc = args['data'][args['hue']]
            if(type(nlpi.pp['stheme']) is str):
                palette = nlpi.pp['stheme']
            else:
                palette = palette_rgb[:len(hueloc.value_counts())]
                
        else:
            hueloc = None
            palette = palette_rgb
            
        sns.set_style("whitegrid", {
            "ytick.major.size": 0.1,
            "ytick.minor.size": 0.05,
            'grid.linestyle': '--'
        })
        
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
        plt.show()
        
    @staticmethod
    def sviolinplot(args:dict):
        
        if(args['hue'] is not None):
            hueloc = args['data'][args['hue']]
            if(type(nlpi.pp['stheme']) is str):
                palette = nlpi.pp['stheme']
            else:
                palette = palette_rgb[:len(hueloc.value_counts())]
                
        else:
            hueloc = None
            palette = palette_rgb
            
        sns.set_style("whitegrid", {
            "ytick.major.size": 0.1,
            "ytick.minor.size": 0.05,
            'grid.linestyle': '--'
        })
            
        sns.violinplot(x=args['x'], 
                       y=args['y'],
                       hue=args['hue'],
                       palette=palette,
                       data=args['data'])
        
        sns.despine(left=True, bottom=True)
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
        
    @staticmethod
    def shistplot(args:dict):
        
        if(args['hue'] is not None):
            hueloc = args['data'][args['hue']]
            if(type(nlpi.pp['stheme']) is str):
                palette = nlpi.pp['stheme']
            else:
                palette = palette_rgb[:len(hueloc.value_counts())]
                
        else:
            hueloc = None
            palette = palette_rgb

        sns.set_style("whitegrid", {
            "ytick.major.size": 0.1,
            "ytick.minor.size": 0.05,
            'grid.linestyle': '--'
        })
        
        # barwidth
        if(args['bw'] is None):
            bw = 'auto'
        else:
            bw = eval(args['bw'])

       # bar width
        if(args['nbins'] is None):
            nbins = 'auto'
        else:
            nbins = eval(args['nbins'])
        
        sns.histplot(x=args['x'], 
                     y=args['y'],
                     hue = args['hue'],
                     bins = bw,
                     nbins = nbins,
                     palette = palette,
                     data=args['data'])
        
        sns.despine(left=True, bottom=True)
        plt.show()
        nlpi.resetpp()
        
    @staticmethod
    def seaborn_kdeplot(args:dict):
            
        if(args['hue'] is not None):
            hueloc = args['data'][args['hue']]
            if(type(nlpi.pp['stheme']) is str):
                palette = nlpi.pp['stheme']
            else:
                palette = palette_rgb[:len(hueloc.value_counts())]
                
        else:
            hueloc = None
            palette = palette_rgb
            
        sns.set_style("whitegrid", {
            "ytick.major.size": 0.1,
            "ytick.minor.size": 0.05,
            'grid.linestyle': '--'
        })            
        
        sns.kdeplot(x=args['x'],
                    y=args['y'],
                    hue = args['hue'],
                    palette=palette,
                    fill=nlpi.pp['fill'],
                    data = args['data'])
        
        sns.despine(left=True, bottom=True)
        plt.show()
        nlpi.resetpp()
        
    @staticmethod
    def split_types(df):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']  
        numeric = df.select_dtypes(include=numerics)
        categorical = df.select_dtypes(exclude=numerics)
        return numeric,categorical
        
    def seaborn_pairplot(self,args:dict):
   
        num,cat = self.split_types(args['data'])
            
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
                               'fill':nlpi.pp['fill']},
                     plot_kws={'edgecolor':nlpi.pp['mec'],
                               'linewidth':nlpi.pp['mew'],
                               'alpha':nlpi.pp['alpha'],
                               's':nlpi.pp['s']})   
        
        sns.despine(left=True, bottom=True)
        plt.show()
        nlpi.resetpp()
        
    # Seaborn Line Plot

    def slineplot(self,args:dict):
    
        if(args['hue'] is not None):
            hueloc = args['data'][args['hue']]
            if(type(nlpi.pp['stheme']) is str):
                palette = nlpi.pp['stheme']
            else:
                palette = palette_rgb[:len(hueloc.value_counts())]
                
        else:
            hueloc = None
            palette = palette_rgb
            
        sns.set_style("whitegrid", {
            "ytick.major.size": 0.1,
            "ytick.minor.size": 0.05,
            'grid.linestyle': '--'
         })

        sns.lineplot(x=args['x'], 
                     y=args['y'],
                     hue=args['hue'],
                     alpha= nlpi.pp['alpha'],
                     linewidth=nlpi.pp['mew'],
                     data=args['data'],
                     palette=palette)
        
        sns.despine(left=True, bottom=True)
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
    