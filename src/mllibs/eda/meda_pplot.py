
from mllibs.nlpi import nlpi
from mllibs.dict_helper import sfp,sfpne,convert_str_to_val
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

class eda_pplot(nlpi):
    
    def __init__(self):
        self.name = 'eda_pplot'  

        path = pkg_resources.resource_filename('mllibs','/eda/meda_pplot.json')
        with open(path, 'r') as f:
            self.json_data = json.load(f)
            self.nlp_config = parse_json(self.json_data)
        
#       default_colors_p = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252'] # my custom (plotly)
        pallete = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
        self.default_colors = pallete

    # set plot parameters

    def set_plotparameters(self):

        if(nlpi.pp['template'] is None):
            self.template = 'plotly_white'
        else:
            self.template = nlpi.pp['template']

        if(nlpi.pp['background'] is None):
            self.background = True
        else:
            self.background = nlpi.pp['background']

        if(nlpi.pp['figsize'] is None):
            self.fheight = None
            self.fwidth = None
        elif(nlpi.pp['figsize'] is not None):
            self.fheight = nlpi.pp['figsize'][0]
            self.fwidth = nlpi.pp['figsize'][1]

        if(nlpi.pp['title'] is None):
            self.title = None
        else:
            self.title = nlpi.pp['title']

    # select activation function
    def sel(self,args:dict):
                
        self.args = args
        select = args['pred_task']
        self.data_name = args['data_name']
        self.subset = args['subset']

        # define plot parameters based on current nlpi.pp parameters
        self.set_plotparameters()

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

    '''

    Activation Functions

    '''

    # plotly basic scatter plot(plscatter)

    def plotly_scatterplot(self,args:dict):

        fig = px.scatter(args['data'],
                         x=args['x'],  # minimum
                         y=args['y'],  # minimum
                         color=args['hue'], # optional
                         facet_col=args['col'], # optional
                         facet_row=args['row'], # optional
                         opacity=args['alpha'], # optional
                         facet_col_wrap=args['col_wrap'], # optional
                         marginal_x = args['marginal_x'], # optional
                         marginal_y = args['marginal_y'], # optional
                         color_discrete_sequence = self.default_colors, # minimum
                         trendline=args['trendline'], # optional
                         template=self.template, 
                         width=self.fwidth,
                         height=self.fheight,
                         title=self.title
                         )

        # Plot Adjustments, in plotly marker size (s), marker edge width (mew) and marker edge color
        # are set in update_traces

        if(args['s'] != 0):
            fig.update_traces(marker={'size':args['s']},selector={'mode':'markers'})
        if(args['mew'] != None):
            fig.update_traces(marker={"line":{'width':args['mew']}},selector={'mode':'markers'})
        if(args['mec'] != None):
            fig.update_traces(marker={"line":{'color':args['mec']}},selector={'mode':'markers'})

        if(self.background is False):
            fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            })

        fig.show()

    # plotly basic box plot (plbox)

    def plotly_boxplot(self,args:dict):

        col_wrap = convert_str_to_val(args,'col_wrap')
        # nbins = convert_str_to_val(args,'nbins')

        fig = px.box(args['data'],
                     x=args['x'],
                     y=args['y'],
                     color=args['hue'],
                     facet_col=args['col'],
                     facet_row=args['row'],
                     facet_col_wrap=col_wrap,
                     color_discrete_sequence = self.default_colors,
                     template=self.template, 
                     width=self.fwidth,
                     height=self.fheight,
                     title=self.title)

        if(nlpi.pp['background'] is False):
            fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            })

        fig.show()
      
    # plotly basic histogram plot (plhist)
    def plotly_histogram(self,args:dict):

        col_wrap = convert_str_to_val(args,'col_wrap')
        nbins = convert_str_to_val(args,'nbins')

        fig = px.histogram(args['data'],
                           x=args['x'],
                           y=args['y'],
                           color=args['hue'],
                           facet_col=args['col'],
                           facet_row=args['row'],
                           facet_col_wrap=col_wrap,
                           nbins=nbins,
                           template=self.template, 
                           width=self.fwidth,
                           height=self.fheight,
                           title=self.title)

        fig.show()

    # plotly basic histogram plot (plhist)
    def plotly_line(self,args:dict):

        col_wrap = convert_str_to_val(args,'col_wrap')

        fig = px.line(args['data'],
                       x=args['x'],
                       y=args['y'],
                       color=args['hue'],
                       facet_col=args['col'],
                       facet_row=args['row'],
                       facet_col_wrap=col_wrap,
                       template=self.template, 
                       width=self.fwidth,
                       height=self.fheight,
                       title=self.title
        )

        fig.show()

    # [plotly] Violin plot (plviolin)

    def plotly_violin(self,args:dict):

        col_wrap = convert_str_to_val(args,'col_wrap')

        fig = px.violin(args['data'],
                       x=args['x'],
                       y=args['y'],
                       color=args['hue'],
                       facet_col=args['col'],
                       facet_row=args['row'],
                       facet_col_wrap=col_wrap,
                       box=True,
                       template=self.template, 
                       width=self.fwidth,
                       height=self.fheight,
                       title=self.title
                       )

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
                     template=self.template, 
                     width=self.fwidth,
                     height=self.fheight,
                     title=self.title
                     )

        fig.show()

    # [plotly] Heatmap (plheatmap)

    def plotly_heatmap(self,args:dict):

        col_wrap = convert_str_to_val('col_wrap')

        fig = px.density_heatmap(args['data'],
                                 x=args['x'],
                                 y=args['y'],
                                 facet_col=args['col'],
                                 facet_row=args['row'],
                                 facet_col_wrap=col_wrap,
                                 template=self.template, 
                                 width=self.fwidth,
                                 height=self.fheight,
                                 title=self.title
                                 )
        fig.show()
    