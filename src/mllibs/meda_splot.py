
from mllibs.nlpi import nlpi
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import warnings; warnings.filterwarnings('ignore')


# Define Palette
def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

palette = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252']
palette_rgb = [hex_to_rgb(x) for x in palette]

'''

Standard seaborn library visualisations


'''


# sample module class structure
class eda_plot(nlpi):
    
    # called in nlpm
    def __init__(self,nlp_config):
        self.name = 'eda_plot'             
        self.nlp_config = nlp_config  
        
    # called in nlpi
    def sel(self,args:dict):
                
        select = args['pred_task']
        
        if(select == 'sscatterplot'):
            self.seaborn_scatterplot(args)
        elif(select =='srelplot'):
            self.seaborn_relplot(args)
        elif(select == 'sboxplot'):
            self.seaborn_boxplot(args)
        elif(select == 'sresidplot'):
            self.seaborn_residplot(args)
        elif(select == 'sviolinplot'):
            self.seaborn_violinplot(args)
        elif(select == 'shistplot'):
            self.seaborn_histplot(args)
        elif(select == 'skdeplot'):
            self.seaborn_kdeplot(args)
        elif(select == 'slmplot'):
            self.seaborn_lmplot(args)
        elif(select == 'spairplot'):
            self.seaborn_pairplot(args)
        elif(select == 'slineplot'):
            self.seaborn_lineplot(args)
        elif(select == 'scorrplot'):
            self.seaborn_heatmap(args)
      
    @staticmethod
    def seaborn_scatterplot(args:dict):
           
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

        sns.scatterplot(x=args['x'], 
                        y=args['y'],
                        hue=args['hue'],
                        alpha= nlpi.pp['alpha'],
                        linewidth=nlpi.pp['mew'],
                        edgecolor=nlpi.pp['mec'],
                        s = nlpi.pp['s'],
                        data=args['data'],
                        palette=palette)
        
        sns.despine(left=True, bottom=True)
        plt.show()
        nlpi.resetpp()
        

    @staticmethod
    def seaborn_lmplot(args:dict):
    
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
            'grid.linestyle': '--'})
        
        sns.lmplot(x=args['x'], 
                   y=args['y'],
                   hue=args['hue'],
                   col=args['col'],
                   row=args['row'],
                   data=args['data'],
                   palette=palette)
        
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
    def seaborn_violinplot(args:dict):
        
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
        
    @staticmethod
    def seaborn_residplot(args:dict):
        sns.residplot(x=args['x'], 
                      y=args['y'],
                      color=nlpi.pp['stheme'][1],
                      data=args['data'])
        
        sns.despine(left=True, bottom=True)
        plt.show()
        
    @staticmethod
    def seaborn_histplot(args:dict):
        
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
        
        # bar width
        if(args['bw'] is None):
            bw = 'auto'
        else:
            bw = eval(args['bw'])
        
        sns.histplot(x=args['x'], 
                     y=args['y'],
                     hue = args['hue'],
                     bins = bw,
                     palette = palette,
                     data=args['data'])
        
        sns.despine(left=True, bottom=True)
        plt.show()
        
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
                    palette=palette,
                    fill=nlpi.pp['fill'],
                    data = args['data'],
                    hue = args['hue'])
        
        sns.despine(left=True, bottom=True)
        plt.show()
        
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
        
        
    @staticmethod
    def seaborn_lineplot(args:dict):
    
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
                
    def seaborn_heatmap(self,args:dict):
        
        if(args['hue'] is not None):
            hueloc = args['data'][args['hue']]
            if(type(nlpi.pp['stheme']) is str):
                palette = nlpi.pp['stheme']
            else:
                palette = palette_rgb[:len(hueloc.value_counts())]
                
        else:
            hueloc = None
            palette = palette_rgb
        
        num,cat = self.split_types(args['data'])
        sns.heatmap(num,cmap=palette,
                    square=True,lw=2,
                    annot=True,cbar=True)    
    
        
corpus_edaplt = OrderedDict({  
                            
                            
                            'sscatterplot':['create seaborn scatter plot',
                                           'make seaborn scatter plot',
                                           'make seaborn scatterplot',
                                           'create seaborn scatterplot',
                                           'create scatterplot',
                                           'make scatterplot'],
                            
                            'srelplot': ['create seaborn relplot',
                                         'create seaborn relplot',
                                         'make seaborn relplot',
                                        'create seaborn relation plot',
                                        'create seaborn relation plot',
                                        'make seaborn relation plot'],
                            
                            'sboxplot': ['create seaborn boxplot',
                                        'create seaborn box plot',
                                        'make seaborn boxplot',
                                         'make seaborn box plot'
                                        ],
                            
                             'sresidplot': ['create residual plot',
                                          'make residual plot',
                                          'create seaborn residual plot',
                                          'make seaborn residual plot'],
                             
                             'sviolinplot' : ['create violin plot',
                                             'create violinplot',
                                             'create seaborn violin plot',
                                             'create seaborn violinplot',
                                             'make violin plot',
                                             'make violinplot'],
                             
                         'shistplot' : ['create histogram plot',
                                        'create seaborn histogram plot',
                                        'make histogram plot',
                                        'create histogram',
                                        'make seaboen histogram plot'],
                             
                            'skdeplot' : ['create kde plot',
                                          'create kernel density estimation plot',
                                          'make kde plot',
                                          'create seaborn kde plot',
                                          'make seaborn kde plot',
                                          'plot seaborn kde figure',
                                          'make seaborn kde figure'],
                             
                             'slmplot' : ['create seaborn regression plot',
                                         'create seaborn regression model',
                                         'create seaborn regression estimation',
                                         'make seaborn regression plot',
                                         'make seaborn regression estimation',
                                         'make seaborn regression model',
                                         'plot seaborn regression model',
                                         'plot seaborn estimation',
                                         'fit seaborn regression estimation plot',
                                         'fit regression estimation plot',
                                         'make seaborn regression estimation figure',
                                         'visualise seaborn regression figure'],
                                          
                                "spairplot":['create seaborn pairplot',
                                        'create seaborn pairplot',
                                        'create bivariate pairplot',
                                        'make bivariate pairplot',
                                        'create multivariate seaborn pairplot',
                                        'make multivariate pairplot'],  
                                                       
                             "slineplot" : ['create seaborn lineplot plot',
                                           'make seaborn line plot',
                                           'make seaborn lineplot',
                                           'create seaborn lineplot',
                                           'create lineplot',
                                           'make lineplot'],
                             
                             'scorrplot' : ['create seaborn heatmap plot',
                                              'make seaborn heatmap plot',
                                              'create heatmap plot',
                                              'make heatmap figure',
                                              'visualise heatmap figure']                               
                            
                            })
                            
info_edaplt = {
             
             'sscatterplot': {'module':'eda_plot',
                              'action':'action',
                              'topic':'topic',
                              'subtopic':'sub topic',
                              'input_format':'pd.DataFrame',
                              'output_format':'figure',
                              'description':'make a scatter plot using seaborn, bivariate or multivariate'},    

             'srelplot':     {'module':'eda_plot',
                              'action':'show plot',
                              'topic':'exploratory data analysis (eda)',
                              'subtopic':'scatter figure',
                              'input_format':'pd.DataFrame',
                              'output_format':'figure',
                              'description':'Create a seaborn relation plot using relplot, we can use columns and rows to show differences in data'},
               
             'sboxplot':     {'module':'eda_plot',
                              'action':'show plot',
                              'topic':'exploratory data analysis (eda)',
                              'subtopic':'scatter figure',
                              'input_format':'pd.DataFrame',
                              'output_format':'figure',
                              'description':'Create a seaborn box plot using boxpot, they show quartiles and outliers'},            
               
              'sresidplot':   {'module':'eda_plot',
                               'action':'show plot',
                               'topic':'exploratory data analysis (eda)',
                               'subtopic':'scatter figure',
                               'input_format':'pd.DataFrame',
                               'output_format':'figure',
                               'description':'Create a seaborn residual plot'},
               
               'sviolinplot':  {'module':'eda_plot',
                                'action':'show plot',
                                'topic':'exploratory data analysis (eda)',
                                'subtopic':'scatter figure',
                                'input_format':'pd.DataFrame',
                                'output_format':'figure',
                                'description':'Create a seaborn violin plot'},       
               
               'shistplot':  {'module':'eda_plot',
                              'action':'show plot',
                              'topic':'exploratory data analysis (eda)',
                              'subtopic':'scatter figure',
                              'input_format':'pd.DataFrame',
                              'output_format':'figure',
                              'description':'Create a seaborn histogram plot'},
               
               'skdeplot':  {'module':'eda_plot',
                              'action':'show plot',
                              'topic':'exploratory data analysis (eda)',
                              'subtopic':'scatter figure',
                              'input_format':'pd.DataFrame',
                              'output_format':'figure',
                              'description':'Create a seaborn kernel density estimation plot (kde)'} ,
               
               'slmplot':  {'module':'eda_plot',
                              'action':'show plot',
                              'topic':'exploratory data analysis (eda)',
                              'subtopic':'scatter figure',
                              'input_format':'pd.DataFrame',
                              'output_format':'figure',
                              'description':'Create a seaborn regression model fit'} ,
        
               'spairplot':  {'module':'eda_plot',
                              'action':'show plot',
                              'topic':'exploratory data analysis (eda)',
                              'subtopic':'scatter figure',
                              'input_format':'pd.DataFrame',
                              'output_format':'figure',
                              'description':'Make a seaborn pairplot using seaborn'},
               
                'slineplot':  {'module':'eda_plot',
                              'action':'show plot',
                              'topic':'exploratory data analysis (eda)',
                              'subtopic':'scatter figure',
                              'input_format':'pd.DataFrame',
                              'output_format':'figure',
                              'description':'Make a seaborn lineplot using seaborn'},
               
                'scorrplot':  {'module':'eda_plot',
                              'action':'show plot',
                              'topic':'exploratory data analysis (eda)',
                              'subtopic':'scatter figure',
                              'input_format':'pd.DataFrame',
                              'output_format':'figure',
                              'description':'Make a seaborn heatmap figure/plot of correlation values'},               
                        
              }
                         
# configuration dictionary (passed in nlpm)
configure_edaplt = {'corpus':corpus_edaplt,'info':info_edaplt}