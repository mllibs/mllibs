
## **Module Group**

`src/eda`[^1]

## **Project Stage ID**

[^1]: Reference to the sub folder in `src`

4[^2]

[^2]: Reference to the machine learning project phase identification defined [here](../../mlproject.md)

## :material-frequently-asked-questions: **Purpose**

The purpose of this module is to provide the user with the ability to utilise the basic visualisation tools provided in the library [seaborn](https://seaborn.pydata.org/)

## :fontawesome-solid-location-arrow: **Location**

Here are the locations of the relevant files associated with the module

<h3>module information:</h3>

`/corpus/meda_splot.json`[^3]

[^3]: [location](../../../src/mllibs/corpus/meda_splot.json) | [github](https://github.com/shtrausslearning/mllibs/blob/main/src/mllibs/corpus/meda_splot.json)

<h3>module activation functions:</h3>

`/src/eda/meda_splot.py`[^4]

[^4]: [location](../../../src/mllibs/eda/meda_splot.py) | [github](https://github.com/shtrausslearning/mllibs/blob/main/src/mllibs/eda/meda_splot.py)

## :material-import: **Requirements**

Module import information

```python
from mllibs.nlpi import nlpi
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import warnings; warnings.filterwarnings('ignore')
from mllibs.nlpm import parse_json
import pkg_resources
import json
```

## :material-selection-drag: **Selection**

Activation functions need to be assigned a unique label. Here's the process of `label` & activation function selection 

```python
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
```

## :octicons-code-16: **Activation Functions**

Here you will find the relevant activation functions available in class `meda_scplot`

### :octicons-file-code-16: `sscatterplot`

<h4>description:</h4>

A Seaborn scatterplot is a type of plot used to visualize the relationship between two variables in a dataset. It is created using the seaborn library in Python and is often used to identify patterns and trends in the data.

The plot shows a scatterplot of the data points, with each point representing a single observation. The x and y axes show the values of the two variables being plotted, and the plot can be customized to show additional information, such as a regression line or confidence intervals.

The Seaborn scatterplot is a useful tool for exploring and visualizing relationships in your data, and can help you to identify any outliers or unusual observations.

<h4>code:</h4>

```python linenums="1"
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
                    alpha = nlpi.pp['alpha'],
                    linewidth=nlpi.pp['mew'],
                    edgecolor=nlpi.pp['mec'],
                    s = nlpi.pp['s'],
                    data=args['data'],
                    palette=palette)
    
    sns.despine(left=True, bottom=True)
    plt.show()
    nlpi.resetpp()
```

### :octicons-file-code-16: `srelplot`

<h4>description:</h4>

A Seaborn relplot is a type of plot used to visualize the relationship between two variables in a dataset. It is created using the seaborn library in Python and is often used to identify patterns and trends in the data.

The plot shows a scatterplot of the data points, with each point representing a single observation. The x and y axes show the values of the two variables being plotted, and the plot can be customized to show additional information, such as a regression line or confidence intervals.

The relplot can also be used to group the data by a categorical variable, creating separate plots for each group. This allows you to compare the relationship between the variables across different groups within the dataset.

Overall, the Seaborn relplot is a powerful tool for exploring and visualizing relationships in your data.

<h4>code:</h4>

```python linenums="1"
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
```
### :octicons-file-code-16: `sboxplot`

<h4>description:</h4>

A Seaborn boxplot is a type of plot used to visualize the distribution of a dataset. It is created using the seaborn library in Python and is often used to identify outliers and compare the distribution of different groups or categories within a dataset.

The plot shows a box that represents the interquartile range (IQR) of the data, which is the range between the 25th and 75th percentiles. The line inside the box represents the median value, while the whiskers extend to show the range of the data, excluding any outliers. Outliers are plotted as individual points beyond the whiskers.

The boxplot can be customized to show additional information, such as the mean value or confidence intervals, and can be grouped by a categorical variable to compare the distribution of different groups within the dataset. By examining the boxplot, you can identify any skewness or asymmetry in the distribution, as well as any extreme values that may need to be addressed.

<h4>code:</h4>

```python linenums="1"
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
```

### :octicons-file-code-16: `sresidplot`

<h4>description:</h4>

A Seaborn residual plot is a type of plot used to visualize the residuals (the difference between the predicted values and the actual values) of a regression model. It is created using the seaborn library in Python and is often used to check whether the assumptions of linear regression are met, such as linearity, homoscedasticity, and normality. 

The plot shows the distribution of the residuals on the y-axis and the predicted values on the x-axis. The residuals are plotted as points with a horizontal line at zero to show the expected value of the residuals if the model is accurate. The plot also includes a fitted line that represents the regression line of the model.

By examining the residual plot, you can identify patterns or trends in the residuals that may indicate that the model is not appropriate for the data or that there are outliers or influential points that need to be addressed.

<h4>code:</h4>

```python linenums="1"
@staticmethod
def seaborn_residplot(args:dict):
    sns.residplot(x=args['x'], 
                  y=args['y'],
                  color=nlpi.pp['stheme'][1],
                  data=args['data'])
    
    sns.despine(left=True, bottom=True)
    plt.show()
```

### :octicons-file-code-16: `sviolinplot`

<h4>description:</h4>

A Seaborn violinplot is a type of plot used to visualize the distribution of a single variable in a dataset. It is created using the seaborn library in Python and is often used to compare the distribution of the variable across different categories or groups.

The plot shows a "violin" shape, which represents the distribution of the data. The width of the violin at any point represents the density of observations at that point, with wider parts indicating more observations. The plot can also show additional information, such as the median and quartiles of the data.

The Seaborn violinplot is a useful tool for exploring and visualizing the distribution of your data, and can help you to identify any differences or similarities between groups or categories.

<h4>code:</h4>

```python linenums="1"
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
    nlpi.resetpp()
```

### :octicons-file-code-16: `shistplot`

<h4>description:</h4>

A Seaborn histplot is a type of plot used to visualize the distribution of a single variable in a dataset. It is created using the seaborn library in Python and is often used to explore the shape of the distribution, as well as any outliers or gaps in the data.

The plot shows a histogram, which is a bar chart that represents the frequency of values in the dataset. The x-axis represents the range of values, and the y-axis represents the frequency of those values. The bars are typically grouped into bins, which represent a range of values.

The Seaborn histplot can also show additional information, such as the density of the data or a kernel density estimate. It is a useful tool for exploring and visualizing the distribution of your data, and can help you to identify any patterns or trends in the data.

<h4>code:</h4>

```python linenums="1"
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
    nlpi.resetpp()
```

### :octicons-file-code-16: `skdeplot`

<h4>description:</h4>

A Seaborn kdeplot is a type of plot used to visualize the distribution of a single variable in a dataset, similar to a histplot. However, instead of using bars to represent the frequency of values, it uses a kernel density estimate (KDE) to create a smooth curve that represents the distribution of the data.

The plot shows the density of the data, with higher peaks indicating where the data is more concentrated. The x-axis represents the range of values, and the y-axis represents the density of those values.

A Seaborn kdeplot can also show additional information, such as a rug plot that indicates the location of each individual data point along the x-axis. It is a useful tool for exploring and visualizing the distribution of your data, and can help you to identify any patterns or trends in the data.

<h4>code:</h4>

```python linenums="1"
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
    nlpi.resetpp()
```

### :octicons-file-code-16: `slmplot`

<h4>description:</h4>

A Seaborn lmplot is a type of plot used to visualize the relationship between two variables in a dataset, typically using a scatter plot with a regression line. It can be used to explore the correlation between two variables and to identify any patterns or trends in the data.

The lmplot function in Seaborn allows you to specify various parameters, such as the x and y variables, the data source, and the type of regression model to use. It also allows you to add additional information to the plot, such as confidence intervals and hue variables that can be used to group the data by a categorical variable.

Overall, Seaborn lmplots are a useful tool for exploring and visualizing the relationship between two variables in a dataset, and can help you to gain insights into the underlying patterns and trends in your data.

<h4>code:</h4>

```python linenums="1"
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
```

### :octicons-file-code-16: `spairplot`

<h4>description:</h4>

A Seaborn pairplot is a type of plot used to visualize the pairwise relationships between variables in a dataset. It creates a grid of scatterplots and histograms, where each variable in the dataset is plotted against every other variable. This allows you to quickly visualize how different variables are related to each other and to identify any patterns or trends in the data.

The pairplot function in Seaborn allows you to specify various parameters, such as the data source, the variables to include, and the type of plot to use for the diagonal (e.g., a histogram or kernel density plot). It also allows you to add additional information to the plot, such as hue variables that can be used to group the data by a categorical variable.

Overall, Seaborn pairplots are a useful tool for exploring and visualizing the relationships between variables in a dataset, and can help you to gain insights into the underlying patterns and trends in your data.

<h4>code:</h4>

```python linenums="1"
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
```

### :octicons-file-code-16: `slineplot`

<h4>description:</h4>

A Seaborn lineplot is a type of plot used to visualize the relationship between two continuous variables. It creates a line chart that displays the trend in the data over time or across some other continuous variable. Lineplots are useful for identifying trends and patterns in data, as well as for comparing the values of different variables over time or across different categories.

The lineplot function in Seaborn allows you to specify various parameters, such as the data source, the variables to include, and the type of aggregation to use (e.g., mean, median, or sum). It also allows you to add additional information to the plot, such as hue variables that can be used to group the data by a categorical variable.

Overall, Seaborn lineplots are a useful tool for exploring and visualizing the relationships between continuous variables in a dataset, and can help you to gain insights into the underlying patterns and trends in your data.

<h4>code:</h4>

```python linenums="1"
def seaborn_lineplot(self,args:dict):

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
```

### :octicons-file-code-16: `slineplot`

<h4>description:</h4>

A Seaborn heatmap is a graphical representation of data that uses color-coded cells to display values in a matrix. Heatmaps are commonly used to visualize the correlation between variables in a dataset, where each cell in the matrix represents the correlation coefficient between two variables. The color of the cell indicates the strength and direction of the correlation, with warmer colors (e.g., red) indicating positive correlations and cooler colors (e.g., blue) indicating negative correlations.

Seaborn heatmaps can be customized with various parameters, such as the color palette, the axis labels, and the size and shape of the plot. They are useful for identifying patterns and trends in large datasets, and can help to highlight areas of high or low correlation between variables. Overall, Seaborn heatmaps are a powerful tool for exploring and visualizing complex data relationships in a clear and intuitive way.

<h4>code:</h4>

```python linenums="1"
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
    
    num,_ = self.split_types(args['data'])
    sns.heatmap(num,cmap=palette,
                square=False,lw=2,
                annot=True,cbar=True)    
                
    plt.show()
    nlpi.resetpp()
```