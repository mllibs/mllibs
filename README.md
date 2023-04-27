![](https://i.imgur.com/y9vUpa0.png)

#### **PROJECT INFORMATION**
![](https://camo.githubusercontent.com/d38e6cc39779250a2835bf8ed3a72d10dbe3b05fa6527baa3f6f1e8e8bd056bf/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f436f64652d507974686f6e2d696e666f726d6174696f6e616c3f7374796c653d666c6174266c6f676f3d707974686f6e266c6f676f436f6c6f723d776869746526636f6c6f723d326262633861) ![](https://badgen.net/badge/status/WIP/blue) 

#### **ABOUT PROJECT**

- **mllibs** is a Machine Learning (ML) library which utilises natural language processing (NLP)
- In this notebook, we look at implementing an **NLP interpreter** module for ML related operations
- Development of such helper modules are motivated by the fact that everyones understanding of coding & subject matter (ML in this case) may be different 
- Often we see people create **functions** and **classes** to simplify the process of achieving something (which is good practice)
- Likewise, **NLP interpreters** follow this trend as well, except, in this case our only inputs for activating certain code is **natural language**
- In python, we can interpret **natural language** in the form of **string** type data

<br>

#### **AUTOMATION EXAMPLES**

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

#### **WHY THIS LIBRARY EXISTS**

A good question to ask ourselves is why would this be needed?

Here are some anwsers:
> - Not everyone level of programming is the same, someone might struggle, whilst others know it quite well
> - The same goes for the topic 'Machine Learning', there are quite a few concepts to remember


#### **Package aims to provide:**
- A userfiendly way introduction to Machine Learning for someone new to the field that have little knowledge of programming

<br>


#### **LIBRARY COMPONENTS**
- `nlp module` - stores all created submodules 
- `nlp interpreter` - main module that interprets user text

#### **KAGGLE NOTEBOOK**

Kaggle is used as a testing platform: **<code>[nlp module for mllibs](https://www.kaggle.com/code/shtrausslearning/nlp-nlp-module-for-mllibs)</code>**

#### **Kaggle** | **Github** version: 
- **<code>[ml-models](https://www.kaggle.com/datasets/shtrausslearning/ml-models)</code>** **0.0.1** | **<code>[mllibs](https://github.com/shtrausslearning/mllibs)</code>** **0.0.1**

#### pypi version:
[![PyPI version](https://badge.fury.io/py/mllibs.svg)](https://badge.fury.io/py/mllibs)

#### **src** content:
- `bl_regressor` - Bayesian Linear Regression Class
- `gmm` - Gaussian Mixture Clustering Class
- `gp_bclassifier` - Gaussian Process Binary Classification Class
- `gp_regressor` - Gaussian Process Regression Class
- `gpr_bclassifier` - Gaussian Process Regression Class (Turned Binary Classifier)
- `kriging_regressor` - Universal Kriging Regression Class

#### Installation:

```python
!pip install mllibs
```

#### **Used in Notebooks (Examples):**
- **[Gaussian Processes | Airfoil Noise Modeling](https://www.kaggle.com/code/shtrausslearning/gaussian-processes-airfoil-noise-modeling)**
- **[Geospatial Data Visualisation | Australia](https://www.kaggle.com/code/shtrausslearning/geospatial-data-visualisation-australia)**
- **[Bayesian Regression | House Price Prediction](https://www.kaggle.com/code/shtrausslearning/bayesian-regression-house-price-prediction)**
- **[Heart Disease | Gaussian Process Models](https://www.kaggle.com/code/shtrausslearning/heart-disease-gaussian-process-models)**
- **[Spectogram Broadband Model & Peak Identifier](https://www.kaggle.com/code/shtrausslearning/spectogram-broadband-model-peak-identifier)**
- **[CFD Trade-Off Study Visualisation | Response Model](https://www.kaggle.com/code/shtrausslearning/cfd-trade-off-study-visualisation-response-model)** 
