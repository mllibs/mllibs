![](https://i.imgur.com/Okdo3qc.jpg)

![](https://camo.githubusercontent.com/d38e6cc39779250a2835bf8ed3a72d10dbe3b05fa6527baa3f6f1e8e8bd056bf/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f436f64652d507974686f6e2d696e666f726d6174696f6e616c3f7374796c653d666c6174266c6f676f3d707974686f6e266c6f676f436f6c6f723d776869746526636f6c6f723d326262633861) ![](https://badgen.net/badge/status/WIP/blue) 

#### **1. ABOUT PROJECT**

***

- **mllibs** is a Machine Learning (ML) library which utilises natural language processing (NLP)
- In this notebook, we look at implementing an **NLP interpreter** module for ML related operations
- Development of such helper modules are motivated by the fact that everyones understanding of coding & subject matter (ML in this case) may be different 
- Often we see people create **functions** and **classes** to simplify the process of achieving something (which is good practice)
- Likewise, **NLP interpreters** follow this trend as well, except, in this case our only inputs for activating certain code is **natural language**
- Using python, we can interpret **natural language** in the form of **string** type data, using **natural langauge interpreters**

<br>

#### **2. CODE AUTOMATION**

***

I'm sure most people are familiar with code automation:

(1) Let's take a look at how we would simplify our code using `functions`

```python
def fib_list(n):
    result = []
    a,b = 0,1
    while a<n:
        result.append(a)
        a,b = b, a + b
    return result

fib_list(5) # [0, 1, 1, 2, 3]
```

(2) Let's take a look at how we wold simplify our code using a `class` structure:

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
fib.get_list() # [0, 1, 1, 2, 3]
```

Such approaches presume we have coding knowledge, our next approach doesn't require such knowledge

(3) Let's take a look how we could simplify this using **language**

```python
input = 'calculate the fibonacci sequence for the value of 5'
nlp_interpreter(input) # [0, 1, 1, 2, 3]
```

<br>

#### **3. LETS LOOK AT AN EXAMPLE**

***

Let's check out one example of how it works; let's visualise some data by requesting:

![](https://i.imgur.com/20f3i1Y.jpg)

Our cell output will be:

![](https://i.imgur.com/4TTAAgp.png)

<br>

#### **4. WHY THIS LIBRARY EXISTS**

***

A good question to ask ourselves is why would this be needed?

Here are some anwsers:
> - Not everyone level of programming is the same, someone might struggle, whilst others know it quite well
> - The same goes for the topic 'Machine Learning', there are quite a few concepts to remember


#### **4. Package aims to provide:**
- A userfiendly way introduction to Machine Learning for someone new to the field that have little knowledge of programming

<br>


#### **5. PROJECT STATUS**

***

- `mllibs` is usable, but still very raw, I'm constantly trying way to improve and clean the code structure
- If you would like to try it out, you can simply fork the notebook **<code>[nlp module for mllibs](https://www.kaggle.com/code/shtrausslearning/nlp-nlp-module-for-mllibs)</code>**

<br>

#### **6. LIBRARY COMPONENTS**

***

`mllibs` consists of two parts:

(1) modules associated with the **interpreter**

- `nlpm` - groups together everything required for the interpreter module `nlpi`
- `nlpi` - main interpreter component module (requires `nlpm` instance)
- `snlpi` - single request interpreter module (uses `nlpi`)
- `mnlpi` - multiple request interpreter module (uses `nlpi`)
- `interface` - interactive module (chat type)

(2) custom added modules, for mllibs these library are associated with **machine learning**

- `loader` - file management module, who's role is to access data, so it can be used (testing module)
- `simple_eda` - exploratory data analysis module (testing module)
- `eda_plot` - exploratory data analysis module associated with static graph plots (utilises seaborn)
- `encoder` - nlp related module for converting text to numeric representation (text encoding)
- `embedding` - nlp related module for generating embeddings for tokenised text data
- `cleantext` - nlp related module for cleaning input text data
- `sklinear` - Linear Regression module (testing module)
- `hf_pipeline` - nlp related module for accessing huggingface pipelines

<br>

#### **7. MODULE COMPONENT STRUCTURE**

***

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

#### **8. CREATING A COLLECTION**

***

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
interpreter = nlpi(collection)
```

class `nlpi` can be used with method `exec` for user input interpretation

```python

interpreter.exec('create a scatterplot using data with x dimension1 y dimension2')

```

