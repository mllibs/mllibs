<p align="center">
    <img src="https://i.imgur.com/XDJsh28.png">
</p>

<h5 align="center">
    
![](https://img.shields.io/badge/python-3.7%2B-fc4c69) ![](https://img.shields.io/badge/pypi-v0.2.0-fc4c69) ![](https://img.shields.io/badge/License-MIT-fc4c69)
![License](https://img.shields.io/github/stars/mllibs?style=social)

</h5>

Update!

### **About mllibs**

Some key points about the library:

- **mllibs** is a Machine Learning (ML) library which utilises natural language processing (NLP)
- Development of such helper modules are motivated by the fact that everyones understanding of coding & subject matter (ML in this case) may be different 
- Often we see people create functions and classes to simplify the process of code automation (which is good practice)
- Likewise, NLP based interpreters follow this trend as well, except, in this case our only inputs for activating certain code is **natural language**
- Using python, we can interpret **natural language** in the form of string type data, using natural langauge interpreters
- **mllibs** aims to provide an automated way to do machine learning using natural language

### **Code Automation**

#### Types of Approaches

There are different ways we can automate code execution:
- The first two (<b>function</b>,<b>class</b>) should be familiar, such approaches presume we have coding knowledge.
- Another approach is to utilise <b>natural language</b> to automate code automation, this method doesn't require any coding knowledge. 

<h4>Function</h4>

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

<h4>Class</h4>

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


<h4>Natural Language</h4>

Another approach, which <b>mllibs</b> uses in natural language based code automation:

```python
input = 'calculate the fibonacci'
         sequence for the value of 5'

nlp_interpreter(input) 
```

All these methods will give the following result:

```
[0, 1, 1, 2, 3]
```

### **Library Contents**

**mllibs** is constantly evolving and has a repository documentation and development page which you can visit, **[mllibs documentation](https://shtrausslearning.github.io/mllibs/about.html)**


### **How to Contribute**

<h4>Predefined Tasks</h4>

I'm constantly looking for people to contribute to the development of the library. I've **[created a page](https://shtrausslearning.github.io/mllibs/group/status.html#task-allocation)** where I set different tasks that you can do and join the **[mllibs group](https://github.com/mllibs)**, if you are interested, please get in touch me on telegram **[shtrauss2](https://t.me/shtrauss2)** or via **[shtrausslearning](https://kaggle.com/shtrausslearning)**

<h4>Our own ideas and contributions</h4>

Here's how you can get started:

1. Fork the repository
2. Create a new branch for your changes
3. Make your changes and commit them
4. Submit a pull request