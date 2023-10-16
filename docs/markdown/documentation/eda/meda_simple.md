## **Module Group**

`src/eda`[^1]

## **Project Stage ID**

[^1]: Reference to the sub folder in `src`

4[^2]

[^2]: Reference to the machine learning project phase identification defined [here](../../mlproject.md)

## :material-frequently-asked-questions: **Purpose**

The purpose of this module is to provide the simple exploratory data analysis operations using `pandas`

## :fontawesome-solid-location-arrow: **Location**

Here are the locations of the relevant files associated with the module

<h4>module information:</h4>

`/corpus/meda_simple.json`[^3]

[^3]: [location](../../../src/mllibs/corpus/meda_simple.json) | [github](https://github.com/shtrausslearning/mllibs/blob/main/src/mllibs/corpus/meda_simple.json)

<h4>module activation functions:</h4>

`/src/eda/meda_simple.py`[^4]

[^4]: [location](../../../src/mllibs/eda/meda_simple.py) | [github](https://github.com/shtrausslearning/mllibs/blob/main/src/mllibs/eda/meda_simple.py)

## :material-import: **Requirements**

Module import information

```python
import numpy as np
import pandas as pd
from collections import OrderedDict
from mllibs.nlpi import nlpi
from mllibs.nlpm import parse_json
import pkg_resources
import json
```

## :material-selection-drag: **Selection**

Activation functions need to be assigned a unique label. Here's the process of `label` & activation function selection 

```python
def sel(self,args:dict):
    
    # activation function class name
    select = args['pred_task'] 
                
    # activate relevant function 
    if(select == 'show_info'):
        self.show_info(args)
    
    if(select == 'show_missing'):
        self.show_missing(args)
        
    if(select == 'show_stats'):
        self.show_statistics(args)
        
    if(select == 'show_dtypes'):
        self.show_dtypes(args)
        
    if(select == 'show_corr'):
        self.show_correlation(args)
```

## :octicons-code-16: **Activation Functions**

Here you will find the relevant activation functions available in class `meda_simple`

### :octicons-file-code-16: `show_info`

<h4>description:</h4>

The `pandas.DataFrame.info()` method is used to print a concise summary of a pandas DataFrame. It provides information such as the number of rows and columns, the data types of each column, the memory usage, and the number of non-null values in each column. This method is useful for quickly understanding the structure and content of a DataFrame, especially when working with large datasets. Additionally, it can help identify missing or null values that may need to be addressed in data cleaning or preprocessing.

<h4>code:</h4>

```python linenums="1"
@staticmethod
def show_info(args:dict):
    print(args['data'].info())
```

### :octicons-file-code-16: `show_missing`

<h4>description:</h4>

The `pandas.DataFrame.info()` method is used to print a concise summary of a pandas DataFrame. It provides information such as the number of rows and columns, the data types of each column, the memory usage, and the number of non-null values in each column. This method is useful for quickly understanding the structure and content of a DataFrame, especially when working with large datasets. Additionally, it can help identify missing or null values that may need to be addressed in data cleaning or preprocessing.

<h4>code:</h4>

```python
@staticmethod
def show_missing(args:dict):
    print(args['data'].isna().sum(axis=0))
```

### :octicons-file-code-16: `show_stats`

<h4>description:</h4>

`pandas.DataFrame.describe()` is a method that provides a summary of the statistical properties of each column in a DataFrame. By default, it calculates the count, mean, standard deviation, minimum, 25th percentile, median (50th percentile), 75th percentile, and maximum for each numeric column. 

<h4>code:</h4>

```python
@staticmethod
def show_statistics(args:dict):
    display(args['data'].describe())
```

### :octicons-file-code-16: `show_dtypes`

<h4>description:</h4>

`pandas.DataFrame.dtypes` is an attribute of a pandas DataFrame that returns the data types of each column in the DataFrame. This attribute is useful for understanding the data types of each column and can be used to convert columns to different data types if necessary.

<h4>code:</h4>

```python
@staticmethod
def show_dtypes(args:dict):
    print(args['data'].dtypes)
```

### :octicons-file-code-16: `show_corr`

<h4>description:</h4>

`pandas.DataFrame.corr()` is a method that calculates the correlation between columns in a DataFrame. Correlation is a statistical measure that indicates the degree to which two variables are related

<h4>code:</h4>

```python
@staticmethod
def show_correlation(args:dict):
    corr_mat = pd.DataFrame(np.round(args['data'].corr(),2),
                         index = list(args['data'].columns),
                         columns = list(args['data'].columns))
    corr_mat = corr_mat.dropna(how='all',axis=0)
    corr_mat = corr_mat.dropna(how='all',axis=1)
    display(corr_mat)
```
