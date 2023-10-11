
<center>
![mllibs](data/images/cnb.png){ width="450" }
</center>

<center>![](https://camo.githubusercontent.com/d38e6cc39779250a2835bf8ed3a72d10dbe3b05fa6527baa3f6f1e8e8bd056bf/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f436f64652d507974686f6e2d696e666f726d6174696f6e616c3f7374796c653d666c6174266c6f676f3d707974686f6e266c6f676f436f6c6f723d776869746526636f6c6f723d326262633861) ![](https://img.shields.io/badge/pypi-v0.1.6-fc4c69) ![](https://img.shields.io/badge/License-MIT-blue) ![](https://static.pepy.tech/badge/mllibs)</center>

### About mllibs

Some key points about the library:

- <code>mllibs</code> is a Machine Learning (ML) library which utilises natural language processing (NLP)
- Development of such helper modules are motivated by the fact that everyones understanding of coding & subject matter (ML in this case) may be different 
- Often we see people create `functions` and `classes` to simplify the process of code automation (which is good practice)
- Likewise, NLP based interpreters follow this trend as well, except, in this case our only inputs for activating certain code is `natural language`
- Using python, we can interpret `natural language` in the form of `string` type data, using `natural langauge interpreters`
- <code>mllibs</code> aims to provide an automated way to do machine learning using natural language

### Code Automation

#### Types of Approaches

There are different ways we can automate code execution:
- The first two (`function`,`class`) should be familiar, such approaches presume we have coding knowledge.
- Another approach is to utilise `natural language` to automate code automation, this method doesn't require any coding knowledge. 

#### (1) Function 

Function based code automation should be very familiar to people who code, we define a function & then simply call the function, entering any relevant input arguments which it requires, in this case `n`

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

#### (2) Class 

Another common approach to automate code is using a class based approach. Utilising `OOP` concepts we can initialise & then call class `methods` in order to automate code:

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

#### (3) Natural Language

Another approach, which `mllibs` uses in natural language based code automation:

```python
input = 'calculate the fibonacci'
         sequence for the value of 5'

nlp_interpreter(input) 
```

All these methods will give the following result:

```
[0, 1, 1, 2, 3]
```

### Library Components

`mllibs` consists of two parts:

(1) modules associated with the **interpreter**

- `nlpm` - groups together everything required for the interpreter module `nlpi`
- `nlpi` - main interpreter component module (requires `nlpm` instance)
- `snlpi` - single request interpreter module (uses `nlpi`)
- `mnlpi` - multiple request interpreter module (uses `nlpi`)
- `interface` - interactive module (chat type)

(2) custom added modules, for mllibs these library are associated with **machine learning** topics

You can check all the activations functions using <code>session.fl()</code> as shown in the sample notebooks in folder <code>examples</code>

### Module Component Structure

Currently new modules can be added using a custom class `sample` and a configuration dictionary 
`configure_sample`

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

### Creating a **`Collection`**

`Modules` which we create need to be assembled together into a `collection`, there are two ways to do this: manually importing and grouping modules or using  <code>interface</code> class

#### **Manually Importing Modules**

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

#### **Import Default Libraries**

The faster way, includes all loaded modules and groups them together for us:

```python
from mllibs.interface import interface
session = interface()
```

### **How to Contibute**

Want to add your own project to our collection? We welcome all contributions, big or small. Here's how you can get started:

1. Fork the repository
2. Create a new branch for your changes
3. Make your changes and commit them
4. Submit a pull request

### **Home**

`mllibs` related posts are available on the **[github pages](https://shtrausslearning.github.io/mllibs/home/)


### **Contact**

Any questions or comments about the above post can be addressed on the :fontawesome-brands-telegram:{ .telegram } **[mldsai-info channel](https://t.me/mldsai_info)** or to me directly :fontawesome-brands-telegram:{ .telegram } **[shtrauss2](https://t.me/shtrauss2)**, on :fontawesome-brands-github:{ .github } **[shtrausslearning](https://github.com/shtrausslearning)** or :fontawesome-brands-kaggle:{ .kaggle} **[shtrausslearning](https://kaggle.com/shtrausslearning)**


