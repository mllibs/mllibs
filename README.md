![](https://i.imgur.com/kCIhZIZ.jpg)

<h5 align="center">
    
![](https://camo.githubusercontent.com/d38e6cc39779250a2835bf8ed3a72d10dbe3b05fa6527baa3f6f1e8e8bd056bf/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f436f64652d507974686f6e2d696e666f726d6174696f6e616c3f7374796c653d666c6174266c6f676f3d707974686f6e266c6f676f436f6c6f723d776869746526636f6c6f723d326262633861) [![](https://img.shields.io/badge/kaggle-v0.1.6-8A2BE2)](https://www.kaggle.com/datasets/shtrausslearning/mllibs) [![](https://img.shields.io/badge/github-v0.1.6-8A2BE2)](https://github.com/shtrausslearning/mllibs) [![](https://img.shields.io/badge/pypi-v0.1.6-8A2BE2)]([[https://github.com/shtrausslearning/mllibs](https://pypi.org/project/mllibs/)](https://pypi.org/project/mllibs/))
[![License](https://img.shields.io/badge/License-MIT-blue)](#license "Go to license section")

</h5>

### 1 || MLLIBS

Some key points about the library:

- <code>mllibs</code> is a Machine Learning (ML) library which utilises natural language processing (NLP)
- Development of such helper modules are motivated by the fact that everyones understanding of coding & subject matter (ML in this case) may be different 
- Often we see people create `functions` and `classes` to simplify the process of code automation (which is good practice)
- Likewise, NLP based interpreters follow this trend as well, except, in this case our only inputs for activating certain code is `natural language`
- Using python, we can interpret `natural language` in the form of `string` type data, using `natural langauge interpreters`
- <code>mllibs</code> aims to provide an automated way to do machine learning using natural language

<br>

### 2 || CODE AUTOMATION

#### TYPES OF AUTOMATION APPROACHES

There are different ways we can automate code execution:
- The first two should be familiar, such approaches presume we have coding knowledge
- If we utilise `language`, we can utilise natural language processing `nlp` in order to interpret the request and activate relevant `activation functions`

<table>
    
<tr>
<td> <code>function</code> </td> <td> <code>class</code> </td> <td> <code>natural language</code> </td>
</tr>

<tr>
<td>

```python
def fib_list(n):
    result = []
    a,b = 0,1
    while a<n:
        result.append(a)
        a,b = b, a + b
    return result

fib_list(5) 
```
    
</td>
<td>
    
```python

class fib_list:
    
    def __init__(self,n):
        self.n = n

    def get_list(self):
        result = []
        a,b = 0,1
        while a<self.n:
            result.append(a)
            a,b = b, a + b
        return result

fib = fib_list(5)
fib.get_list()
```

</td>

<td>

```python
input = 'calculate the fibonacci
         sequence for the value of 5'

nlp_interpreter(input) 
```
    
</td>

<tr>
<td> <code>[0, 1, 1, 2, 3]</code> </td> <td> <code>[0, 1, 1, 2, 3]</code> </td> <td> <code>[0, 1, 1, 2, 3]</code> </td>
</tr>

</tr>
</table>

#### **SAMPLE FROM MLLIBS**

Let's check out one example of how it works; let's visualise some data by requesting:

![](https://i.imgur.com/20f3i1Y.jpg)

Our cell output will be:

![](https://i.imgur.com/4TTAAgp.png)

<br>

### 3 || LIBRARY COMPONENTS

`mllibs` consists of two parts:

(1) modules associated with the **interpreter**

- `nlpm` - groups together everything required for the interpreter module `nlpi`
- `nlpi` - main interpreter component module (requires `nlpm` instance)
- `snlpi` - single request interpreter module (uses `nlpi`)
- `mnlpi` - multiple request interpreter module (uses `nlpi`)
- `interface` - interactive module (chat type)

(2) custom added modules, for mllibs these library are associated with **machine learning**

You can check all the activations functions using <code>session.fl()</code> as shown in the sample notebooks in folder <code>examples</code>

<br>

### 4 || MODULE COMPONENT STRUCTURE

Currently new modules can be added using a custom class `sample` and a configuration dictionary `configure_sample`

```python

# sample module class structure
class sample(nlpi):
    
    # called in nlpm
    def __init__(self,nlp_config):
        self.name = 'sample'             # unique module name identifier (used in nlpm/nlpi)
        self.nlp_config = nlp_config  # text based info related to module (used in nlpm/nlpi)
        
    # called in nlpi
    def sel(self,args:dict):
        
        self.select = args['pred_task']
        self.args = args
        
        if(self.select == 'function'):
            self.function(self.args)
        
    # use standard or static methods
        
    def function(self,args:dict):
        pass
        
    @staticmethod
    def function(args:dict):
        pass
    

corpus_sample = OrderedDict({"function":['task']}
info_sample = {'function': {'module':'sample',
                            'action':'action',
                            'topic':'topic',
                            'subtopic':'sub topic',
                            'input_format':'input format for data',
                            'output_format':'output format for data',
                            'description':'write description'}}
                         
# configuration dictionary (passed in nlpm)
configure_sample = {'corpus':corpus_sample,'info':info_sample}

```

<br>

### 5 || CREATING A COLLECTION

There are two ways to start an interpreter session, manually importing and grouping modules or using  <code>interface</code> class

#### **FIRST APPROACH**

First we need to combine all our module components together, this will link all passed modules together

```python

collection = nlpm()
collection.load([loader(configure_loader),
                 simple_eda(configure_eda),
                 encoder(configure_nlpencoder),
                 embedding(configure_nlpembed),
                 cleantext(configure_nlptxtclean),
                 sklinear(configure_sklinear),
                 hf_pipeline(configure_hfpipe),
                 eda_plot(configure_edaplt)])
                 
```

Then we need to train `interpreter` models

```python
collection.train()
```

Lastly, pass the collection of modules (`nlpm` instance) to the interpreter `nlpi` 

```python
session = nlpi(collection)
```

class `nlpi` can be used with method `exec` for user input interpretation

```python

session.exec('create a scatterplot using data with x dimension1 y dimension2')

```

#### **SECOND APPROACH**

The faster way, includes all loaded modules and groups them together for us:

```python
from mllibs.interface import interface
session = interface()
```

<br>

### 6 || SAMPLE NOTEBOOKS

Here are some notebooks that will help you familiarise yourself with the library:

- **[Sample EDA notebook](https://www.kaggle.com/code/shtrausslearning/mllibs-sample-eda-notebook)** (exploratory data analysis options)
- **[Convert text into numeric data ](https://www.kaggle.com/code/shtrausslearning/mllibs-text-to-numeric-representation)** (change text data into numerical representation)
- **[Normalise text](https://www.kaggle.com/code/shtrausslearning/mllibs-text-normalisation)** (Text cleaning preprocessing)
- **[Model Training & Evaluation](https://www.kaggle.com/code/shtrausslearning/mllibs-model-training-evaluation)** (Training and Prediction using supervised learning models)
- **[Dimension Reduction](https://www.kaggle.com/code/shtrausslearning/mllibs-dimensionality-reduction)** (Reducing the size of the feature matrix using unsupervised learning models)

These notebooks can also be found in folder <code>examples</code> and the [kaggle dataset](https://www.kaggle.com/datasets/shtrausslearning/mllibs)

