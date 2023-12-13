
from mllibs.nlpi import nlpi
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import warnings; warnings.filterwarnings('ignore')
from mllibs.nlpm import parse_json
import pkg_resources
import json


'''

Standard Plotly Visualisations (plotly express)

'''

class eda_psplot(nlpi):
    
    def __init__(self):
        self.name = 'eda_pplot'  

        path = pkg_resources.resource_filename('mllibs','/eda/meda_psplot.json')
        with open(path, 'r') as f:
            self.json_data = json.load(f)
            self.nlp_config = parse_json(self.json_data)
        
    # select activation function
    def sel(self,args:dict):

        # define plotly defaults for nlpi.pp
        if(nlpi.pp['template'] is None):
            nlpi.pp['template'] = 'plotly_white'
        if(nlpi.pp['background'] is None):
            nlpi.pp['background'] = True
                
        self.args = args
        select = args['pred_task']
        self.data_name = args['data_name']
        self.subset = args['subset']
        print('subset',self.subset)
        
        if(select == 'plscatter'):
            self.plotly_scatterplot(args)
        elif(select == 'plbox'):
            self.plotly_boxplot(args)
        elif(select == 'plhist'):
            self.plotly_histogram(args)
        elif(select == 'plline'):
            self.plotly_line(args)
        elif(select == 'plviolin'):
            self.plotly_violin(args)
        elif(select == 'plbarplot'):
            self.plotly_bar(args)
        elif(select == 'plheatmap'):
            self.plotly_heatmap(args)

    # for converting numeric text into int/
    # eg. for when token contains numerical value for PARAM or PP

    def convert_str(self,key:str):
        try:
            try:
                # if args[key] is a string
                val = eval(self.args[key])
            except:
                # else just a value
                val = self.args[key]
        except:
            val = None
        return val

    '''

    Activation Functions

    '''

    # plotly basic scatter plot(plscatter)
    def plotly_scatterplot(self,args:dict):

        fig = px.scatter(args['data'],
                         x=args['x'],
                         y=args['y'],
                         color=args['hue'],
                         facet_col=args['col'],
                         facet_row=args['row'],
                         opacity=nlpi.pp['alpha'],
                         facet_col_wrap=args['col_wrap'],
                         template=nlpi.pp['template'],
                         width=nlpi.pp['figsize'][0],
                         height=nlpi.pp['figsize'][1],
                         title=nlpi.pp['title'])

        fig.update_traces(marker={'size':nlpi.pp['s'],"line":{'width':nlpi.pp['mew'],'color':nlpi.pp['mec']}},
                          selector={'mode':'markers'})

        if(nlpi.pp['background'] is False):
            fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            })

        fig.show()

    # plotly basic box plot (plbox)
    def plotly_boxplot(self,args:dict):

        col_wrap = self.convert_str('col_wrap')
        nbins = self.convert_str('nbins')

        fig = px.box(args['data'],
                     x=args['x'],
                     y=args['y'],
                     color=args['hue'],
                     nbins=nbins,
                     facet_col=args['col'],
                     facet_row=args['row'],
                     facet_col_wrap=col_wrap,
                     template=nlpi.pp['template'],
                     width=nlpi.pp['figsize'][0],
                     height=nlpi.pp['figsize'][1],
                     title=nlpi.pp['title'])

        fig.show()
      
    # plotly basic histogram plot (plhist)
    def plotly_histogram(self,args:dict):

        col_wrap = self.convert_str('col_wrap')
        nbins = self.convert_str('nbins')

        fig = px.histogram(args['data'],
                           x=args['x'],
                           y=args['y'],
                           color=args['hue'],
                           facet_col=args['col'],
                           facet_row=args['row'],
                           facet_col_wrap=col_wrap,
                           nbins=nbins,
                           template=nlpi.pp['template'],
                           width=nlpi.pp['figsize'][0],
                           height=nlpi.pp['figsize'][1],
                           title=nlpi.pp['title'])

        fig.show()

    # plotly basic histogram plot (plhist)
    def plotly_line(self,args:dict):

        col_wrap = self.convert_str('col_wrap')

        fig = px.line(args['data'],
                       x=args['x'],
                       y=args['y'],
                       color=args['hue'],
                       facet_col=args['col'],
                       facet_row=args['row'],
                       facet_col_wrap=col_wrap,
                       template=nlpi.pp['template'],
                       width=nlpi.pp['figsize'][0],
                       height=nlpi.pp['figsize'][1],
                       title=nlpi.pp['title'])

        fig.show()

    # [plotly] Violin plot (plviolin)

    def plotly_violin(self,args:dict):

        col_wrap = self.convert_str('col_wrap')

        fig = px.violin(args['data'],
                       x=args['x'],
                       y=args['y'],
                       color=args['hue'],
                       facet_col=args['col'],
                       facet_row=args['row'],
                       facet_col_wrap=col_wrap,
                       box=True,
                       template=nlpi.pp['template'],
                       width=nlpi.pp['figsize'][0],
                       height=nlpi.pp['figsize'][1],
                       title=nlpi.pp['title'])

        fig.show()

    # [plotly] Bar Plot (plbarplot)

    def plotly_bar(self,args:dict):

        fig = px.bar(args['data'],
                     x=args['x'],
                     y=args['y'],
                     color=args['hue'],
                     facet_col=args['col'],
                     facet_row=args['row'],
                     facet_col_wrap=col_wrap,
                     template=nlpi.pp['template'],
                     width=nlpi.pp['figsize'][0],
                     height=nlpi.pp['figsize'][1],
                     title=nlpi.pp['title'])

        fig.show()

    # [plotly] Heatmap (plheatmap)

    def plotly_heatmap(self,args:dict):

        col_wrap = self.convert_str('col_wrap')

        fig = px.density_heatmap(args['data'],
                                 x=args['x'],
                                 y=args['y'],
                                 facet_col=args['col'],
                                 facet_row=args['row'],
                                 facet_col_wrap=col_wrap,
                                 template=nlpi.pp['template'],
                                 width=nlpi.pp['figsize'][0],
                                 height=nlpi.pp['figsize'][1],
                                 title=nlpi.pp['title'])

        fig.show()
    