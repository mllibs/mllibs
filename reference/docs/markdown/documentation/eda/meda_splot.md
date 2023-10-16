
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

### module information:

`/corpus/meda_splot.json`[^3]

[^3]: [location](../../../src/mllibs/corpus/meda_splot.json) | [github](https://github.com/shtrausslearning/mllibs/blob/main/src/mllibs/corpus/meda_splot.json)

### module activation functions

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

#### description:

A Seaborn scatterplot is a type of plot used to visualize the relationship between two variables in a dataset. It is created using the seaborn library in Python and is often used to identify patterns and trends in the data. The plot shows a scatterplot of the data points, with each point representing a single observation. The x and y axes show the values of the two variables being plotted, and the plot can be customized to show additional information, such as a regression line or confidence intervals. The Seaborn scatterplot is a useful tool for exploring and visualizing relationships in your data, and can help you to identify any outliers or unusual observations.

#### code:

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

#### description:

A Seaborn relplot is a type of plot used to visualize the relationship between two variables in a dataset. It is created using the seaborn library in Python and is often used to identify patterns and trends in the data. The plot shows a scatterplot of the data points, with each point representing a single observation. The x and y axes show the values of the two variables being plotted, and the plot can be customized to show additional information, such as a regression line or confidence intervals. The relplot can also be used to group the data by a categorical variable, creating separate plots for each group. This allows you to compare the relationship between the variables across different groups within the dataset. Overall, the Seaborn relplot is a powerful tool for exploring and visualizing relationships in your data.

#### code:

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

#### description:

A Seaborn boxplot is a type of plot used to visualize the distribution of a dataset. It is created using the seaborn library in Python and is often used to identify outliers and compare the distribution of different groups or categories within a dataset.

The plot shows a box that represents the interquartile range (IQR) of the data, which is the range between the 25th and 75th percentiles. The line inside the box represents the median value, while the whiskers extend to show the range of the data, excluding any outliers. Outliers are plotted as individual points beyond the whiskers.

The boxplot can be customized to show additional information, such as the mean value or confidence intervals, and can be grouped by a categorical variable to compare the distribution of different groups within the dataset. By examining the boxplot, you can identify any skewness or asymmetry in the distribution, as well as any extreme values that may need to be addressed.

#### code:

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

#### description:

A Seaborn residual plot is a type of plot used to visualize the residuals (the difference between the predicted values and the actual values) of a regression model. It is created using the seaborn library in Python and is often used to check whether the assumptions of linear regression are met, such as linearity, homoscedasticity, and normality. The plot shows the distribution of the residuals on the y-axis and the predicted values on the x-axis. The residuals are plotted as points with a horizontal line at zero to show the expected value of the residuals if the model is accurate. The plot also includes a fitted line that represents the regression line of the model. By examining the residual plot, you can identify patterns or trends in the residuals that may indicate that the model is not appropriate for the data or that there are outliers or influential points that need to be addressed.

#### code:

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

