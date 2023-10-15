var __index = {"config":{"lang":["en"],"separator":"[\\s\\-]+","pipeline":["stopWordFilter"]},"docs":[{"location":"home.html","title":"Home","text":""},{"location":"home.html#about-mllibs","title":"About mllibs","text":"<p>Some key points about the library:</p> <ul> <li><code>mllibs</code> is a Machine Learning (ML) library which utilises natural language processing (NLP)</li> <li>Development of such helper modules are motivated by the fact that everyones understanding of coding &amp; subject matter (ML in this case) may be different </li> <li>Often we see people create <code>functions</code> and <code>classes</code> to simplify the process of code automation (which is good practice)</li> <li>Likewise, NLP based interpreters follow this trend as well, except, in this case our only inputs for activating certain code is <code>natural language</code></li> <li>Using python, we can interpret <code>natural language</code> in the form of <code>string</code> type data, using <code>natural langauge interpreters</code></li> <li><code>mllibs</code> aims to provide an automated way to do machine learning using natural language</li> </ul>"},{"location":"home.html#code-automation","title":"Code Automation","text":""},{"location":"home.html#types-of-approaches","title":"Types of Approaches","text":"<p>There are different ways we can automate code execution: - The first two (<code>function</code>,<code>class</code>) should be familiar, such approaches presume we have coding knowledge. - Another approach is to utilise <code>natural language</code> to automate code automation, this method doesn't require any coding knowledge. </p>"},{"location":"home.html#1-function","title":"(1) Function","text":"<p>Function based code automation should be very familiar to people who code, we define a function &amp; then simply call the function, entering any relevant input arguments which it requires, in this case <code>n</code></p> <pre><code>def fib_list(n):\n    result = []\n    a,b = 0,1\n    while a&lt;n:\n        result.append(a)\n        a,b = b, a + b\n    return result\n\nfib_list(5) \n</code></pre>"},{"location":"home.html#2-class","title":"(2) Class","text":"<p>Another common approach to automate code is using a class based approach. Utilising <code>OOP</code> concepts we can initialise &amp; then call class <code>methods</code> in order to automate code:</p> <pre><code>class fib_list:\n\n    def __init__(self,n):\n        self.n = n\n\n    def get_list(self):\n        result = []\n        a,b = 0,1\n        while a&lt;self.n:\n            result.append(a)\n            a,b = b, a + b\n        return result\n\nfib = fib_list(5)\nfib.get_list()\n</code></pre>"},{"location":"home.html#3-natural-language","title":"(3) Natural Language","text":"<p>Another approach, which <code>mllibs</code> uses in natural language based code automation:</p> <pre><code>input = 'calculate the fibonacci'\n         sequence for the value of 5'\n\nnlp_interpreter(input) \n</code></pre> <p>All these methods will give the following result:</p> <pre><code>[0, 1, 1, 2, 3]\n</code></pre>"},{"location":"home.html#library-components","title":"Library Components","text":"<p><code>mllibs</code> consists of two parts:</p> <p>(1) modules associated with the interpreter</p> <ul> <li><code>nlpm</code> - groups together everything required for the interpreter module <code>nlpi</code></li> <li><code>nlpi</code> - main interpreter component module (requires <code>nlpm</code> instance)</li> <li><code>snlpi</code> - single request interpreter module (uses <code>nlpi</code>)</li> <li><code>mnlpi</code> - multiple request interpreter module (uses <code>nlpi</code>)</li> <li><code>interface</code> - interactive module (chat type)</li> </ul> <p>(2) custom added modules, for mllibs these library are associated with machine learning topics</p> <p>You can check all the activations functions using <code>session.fl()</code> as shown in the sample notebooks in folder <code>examples</code></p>"},{"location":"home.html#module-component-structure","title":"Module Component Structure","text":"<p>Currently new modules can be added using a custom class <code>sample</code> and a configuration dictionary  <code>configure_sample</code></p> <pre><code># sample module class structure\nclass sample(nlpi):\n\n    # called in nlpm\n    def __init__(self,nlp_config):\n        self.name = 'sample'             # unique module name identifier (used in nlpm/nlpi)\n        self.nlp_config = nlp_config  # text based info related to module (used in nlpm/nlpi)\n\n    # called in nlpi\n    def sel(self,args:dict):\n\n        self.select = args['pred_task']\n        self.args = args\n\n        if(self.select == 'function'):\n            self.function(self.args)\n\n    # use standard or static methods\n\n    def function(self,args:dict):\n        pass\n\n    @staticmethod\n    def function(args:dict):\n        pass\n\n\ncorpus_sample = OrderedDict({\"function\":['task']}\ninfo_sample = {'function': {'module':'sample',\n                            'action':'action',\n                            'topic':'topic',\n                            'subtopic':'sub topic',\n                            'input_format':'input format for data',\n                            'output_format':'output format for data',\n                            'description':'write description'}}\n\n# configuration dictionary (passed in nlpm)\nconfigure_sample = {'corpus':corpus_sample,'info':info_sample}\n</code></pre>"},{"location":"home.html#creating-a-collection","title":"Creating a <code>Collection</code>","text":"<p><code>Modules</code> which we create need to be assembled together into a <code>collection</code>, there are two ways to do this: manually importing and grouping modules or using  <code>interface</code> class</p>"},{"location":"home.html#manually-importing-modules","title":"Manually Importing Modules","text":"<p>First we need to combine all our module components together, this will link all passed modules together</p> <pre><code>collection = nlpm()\ncollection.load([loader(configure_loader),\n                 simple_eda(configure_eda),\n                 encoder(configure_nlpencoder),\n                 embedding(configure_nlpembed),\n                 cleantext(configure_nlptxtclean),\n                 sklinear(configure_sklinear),\n                 hf_pipeline(configure_hfpipe),\n                 eda_plot(configure_edaplt)])  \n</code></pre> <p>Then we need to train <code>interpreter</code> models</p> <pre><code>collection.train()\n</code></pre> <p>Lastly, pass the collection of modules (<code>nlpm</code> instance) to the interpreter <code>nlpi</code> </p> <pre><code>session = nlpi(collection)\n</code></pre> <p>class <code>nlpi</code> can be used with method <code>exec</code> for user input interpretation</p> <pre><code>session.exec('create a scatterplot using data with x dimension1 y dimension2')\n</code></pre>"},{"location":"home.html#import-default-libraries","title":"Import Default Libraries","text":"<p>The faster way, includes all loaded modules and groups them together for us:</p> <pre><code>from mllibs.interface import interface\nsession = interface()\n</code></pre>"},{"location":"home.html#how-to-contibute","title":"How to Contibute","text":"<p>Want to add your own project to our collection? We welcome all contributions, big or small. Here's how you can get started:</p> <ol> <li>Fork the repository</li> <li>Create a new branch for your changes</li> <li>Make your changes and commit them</li> <li>Submit a pull request</li> </ol>"},{"location":"home.html#contact","title":"Contact","text":"<p>Thank you for reading!</p> <p>Any questions or comments about the above post can be addressed on the  mldsai-info channel or to me directly  shtrauss2, on  shtrausslearning or  shtrausslearning</p>"},{"location":"install.html","title":"Installation","text":""},{"location":"Documentation/nlpi.html","title":"nlpi class","text":""},{"location":"Documentation/nlpm.html","title":"nlpm class","text":""},{"location":"Group/meetups.html","title":"Meetups","text":""},{"location":"Group/meetups.html#group-meetings","title":"Group Meetings","text":"<p>Here's a collection of data from group meetups</p>"},{"location":"Group/meetups.html#-project-introduction","title":"\u276f\u276f project introduction","text":"<p> Slides</p> <p> Oct 15, 2023</p> <ul> <li>Discussion of project</li> <li>Project objectives</li> <li>mllibs internals</li> <li>What subtasks exist</li> <li>Distribution of roles</li> </ul> <p>Duration: 1 Hour</p>"},{"location":"Group/meetups.html#individual-meetings","title":"Individual Meetings","text":"<p>A collections of all individual meetings</p>"},{"location":"Group/members.html","title":"Members","text":""},{"location":"Group/status.html","title":"Statuses","text":""},{"location":"Modules/adding.html","title":"Adding New Modules","text":""},{"location":"Modules/grouping.html","title":"Grouping All Modules","text":""},{"location":"Modules/list.html","title":"Available Modules","text":""},{"location":"Tutorials/overview.html","title":"Overview","text":""}]}