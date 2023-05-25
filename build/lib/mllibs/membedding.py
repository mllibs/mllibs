from keras.preprocessing import text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams
from tensorflow.keras.layers import Dot, Dense, Reshape, Embedding
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
import torch
import numpy as np
import pandas as pd
from torch.optim import Adam
import torch.nn as nn
from gensim.models import word2vec as w2v
from mllibs.nlpi import nlpi
from collections import OrderedDict
from nltk.tokenize import word_tokenize
import nltk
import re

'''

Embedding Generation Only

'''

class embedding(nlpi):
    
    def __init__(self,nlp_config):
        self.name = 'nlp_embedding'
        self.nlp_config = nlp_config 
        self.select = None
        self.data = None
        self.args = None
   
    # describe contents of class

    def sel(self,args:dict):
        
        self.select = args['pred_task']
        self.data = args['data']
        self.args = args    
                
        ''' select appropriate predicted method '''
        
        if(self.select == 'embed_cbow'):
            self.cbow(self.data,self.args)
        elif(self.select == 'embed_sg'):
            self.sg(self.data,self.args)
        elif(self.select ==  'w2v'):
            self.word2vec(self.data,self.args)
            
    # one hot encode dataframe
            
    def cbow(self,data:list,args):
        
        data = data[0]
        tokens = word_tokenize(data)
        token_set = set(tokens) # create all unique tokens
        
        # give unique identifier to each unique token
        word2id = {word:idx for idx,word in enumerate(token_set)} 
        id2word = {idx:word for idx,word in enumerate(token_set)}
        
        if(args['dim'] is not None):
            embeddings = eval(args['dim'])
        else:
            embeddings = 5
        
        if(args['epoch'] is not None):
            epochs = eval(args['epoch'])
        else:
            epochs = 100
            
        if(args['window'] is not None):
            window = eval(args['window'])
        else:
            window = 2
            
        if(args['lr'] is not None):
            lr = eval(args['lr'])
        else:
            lr = 0.001

        # print(token_set) # vocabulary
        vocab_size = len(token_set)  # size of vocabulary

        def context_vector(tokens:list):
            # list of values for each token
            val_context = [word2id[word] for word in tokens] 
            return val_context

        context_pairs = []

        # loop through all possible cases 
        for i in range(window,len(tokens) - window):

            context = []

            # words to the left
            for j in range(-window,0):
                context.append(tokens[i+j])

            # words to the right
            for j in range(1,window+1):
                context.append(tokens[i+j])

            context_pairs.append((context,tokens[i]))

        # sample tensor conversion
        for context,target in context_pairs:
            X = torch.tensor(context_vector(context))
            y = torch.tensor(word2id[target])

        class CBOW(torch.nn.Module):

            def __init__(self,vocab_size,embed_dim):
                super(CBOW,self).__init__()

                self.embedding = nn.Embedding(vocab_size,embed_dim)
                self.linear = nn.Linear(embed_dim,vocab_size)
                self.active = nn.LogSoftmax(dim=-1)

            def forward(self,x):
                x = sum(self.embedding(x)).view(1,-1)
                x = self.linear(x)
                x = self.active(x)
                return x

        model = CBOW(vocab_size,embeddings)    
        criterion = nn.NLLLoss()
        optimiser = Adam(model.parameters(),lr=lr)

        # training loop

        lst_loss = []
        for epoch in range(epochs):

            loss = 0.0
            for context,target in context_pairs:

                X = torch.tensor(context_vector(context))
                y = torch.tensor([word2id[target]])        

                y_pred = model(X)
                loss += criterion(y_pred,y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            lst_loss.append(float(loss.detach().numpy()))

        print(f'loss: {lst_loss[-1]}')
        embeds = list(model.parameters())[0].detach().numpy()
        nlpi.memory_output.append(pd.DataFrame(embeds,index=id2word.values())) 
            
        
    def sg(self,corpus:list,args):
        
        tokeniser = Tokenizer()  # tokeniser initialisation
        tokeniser.fit_on_texts(corpus)  # fit tokeniser on corpus (list of strings)
        vocab_size = len(tokeniser.word_index) + 1

        # tokeniser.word_index - unique words (word,index) dictionary
        # text.text_to_word_sequence - tokenise string
        # text.text_to_sequences - tokenised numerisation

        word2id = tokeniser.word_index # tokens to id
        id2word = {v:k for k, v in word2id.items()} # id to token
        
        if(args['dim'] is not None):
            embed_size = eval(args['dim'])
        else:
            embed_size = 5
        
        if(args['epoch'] is not None):
            epochs = eval(args['epoch'])
        else:
            epochs = 50
            
        if(args['window'] is not None):
            window = eval(args['window'])
        else:
            window = 2
            
        if(args['lr'] is not None):
            lr = eval(args['lr'])
        else:
            lr = 0.001

        # tokenise and convert token to unique number id
        tokens = [[w for w in text.text_to_word_sequence(doc)] for doc in corpus]
        numerical_id = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in corpus]

        ''' Model'''
        
        # word
        word_model = Sequential()
        word_model.add(Embedding(vocab_size, embed_size,input_length=1))
        word_model.add(Reshape((embed_size, )))      # [1,embed_dim] -> [embed_dim]

        # context 
        context_model = Sequential()
        context_model.add(Embedding(vocab_size, embed_size,input_length=1))
        context_model.add(Reshape((embed_size,)))

        # dot product of both embed vectors
        model_arch = Dot(axes=1)([word_model.output, context_model.output]) 
        model_arch = Dense(1,activation="sigmoid")(model_arch)

        model = Model([word_model.input,
                       context_model.input], model_arch)

        optimiser = RMSprop(learning_rate=lr)
        model.compile(loss="mean_squared_error",
                      optimizer=optimiser)
        
        ''' Train Model '''

        lst_loss = []
        for epoch in range(epochs):

            loss = 0.0

            # Enumerate over tokenised text
            for i, doc in enumerate(tokeniser.texts_to_sequences(corpus)):

                # create training samples
                # data - list of [word,context] , label (next to one another)

                data, labels = skipgrams(sequence=doc,   
                                         vocabulary_size=vocab_size, 
                                         window_size=window,
                                         shuffle=True)

                x = [np.array(x) for x in zip(*data)] # word, context vectors 
                y = np.array(labels, dtype=np.int32)  # label (words are next to each other)

                if x:
                    loss += model.train_on_batch(x, y)

            lst_loss.append(loss)
            
        word_embed_layer = model.layers[2]
        word_embed_layer.get_weights()[0].shape
        weights = word_embed_layer.get_weights()[0][1:]
        
        # save embedding values
        nlpi.memory_output.append(pd.DataFrame(weights, index=id2word.values()))
     
    # Word2Vec Embedding Generation
    
    def word2vec(self,data:list,args):
    
        corpus = pd.Series(data)
    
        wpt = nltk.WordPunctTokenizer()
        stop_words = nltk.corpus.stopwords.words('english')
  
        # normalise text
        def normalise(doc):
            doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
            doc = doc.lower()
            doc = doc.strip()
            tokens = wpt.tokenize(doc)
            filtered_tokens = [token for token in tokens if token not in stop_words]
            doc = ' '.join(filtered_tokens)
            return doc
        
        normalize_corpus = np.vectorize(normalise)
        norm_corpus = normalize_corpus(corpus)
    
        # Tokenize corpus
        wpt = nltk.WordPunctTokenizer()
        tokenized_corpus = [wpt.tokenize(document) for document in norm_corpus]

        print('First Tokenised corpus:\n')
        print(tokenized_corpus[0])
    
        # Set Model Parametere                                                                      
        min_word_count = 1           # Minimum word count                        
        sample = 1e-3                # Downsample setting for frequent words
        
        
        if(args['dim'] is not None):
            vector_size = eval(args['dim'])
        else:
            vector_size = 5
            
        if(args['window'] is not None):
            window = eval(args['window'])
        else:
            window = 4
            
        if(args['epoch'] is not None):
            epoch = eval(args['epoch'])
        else:
            epoch = 50
            
        if(args['lr'] is not None):
            alpha = eval(args['lr'])
        else:
            alpha = 0.025
            
        print(f'Epochs: {epoch}')
        print(f'Window: {window}')
        print(f'Vector Size: {vector_size}')

        # Word2Vec Model
        w2v_model = w2v.Word2Vec(tokenized_corpus, 
                         vector_size=vector_size, 
                         window=window,  # context window
                         min_count=min_word_count,
                         sample=sample, 
                         alpha=alpha,
                         epochs=epoch)   
        
        
        vocab_len = len(w2v_model.wv)
        print(f'Vocabulary size: {vocab_len}')

        print('First 10 words in vocabulary:')
        print(w2v_model.wv.index_to_key[:10])
        
        np_list = []
        for word in w2v_model.wv.index_to_key:
            np_list.append(w2v_model.wv[word])
    
        # Calculate mean array of selected document words
        X = pd.DataFrame(np.stack(np_list).T,columns = w2v_model.wv.index_to_key).T
        nlpi.memory_output.append(X)

        
        
dict_nlpembed = {'embed_cbow':  ['cbow embeddings',
                                 'cbow embedding',
                                 'continuous bag of words embedding',
                                 'continuous bag of words embeddings',
                                 'embedding with cbow approach'],
                 
                 'embed_sg'  :  ['skip gram embeddings',
                                 'skip gram embedding',
                                 'sg embeddings',
                                 'sg embedding',
                                 'skip gram',
                                 'embedding with sg',
                                 'embeddings with sg',
                                 'generate skip gram embeddings',
                                 'generate skip gram embedding'],
                  
                 'w2v': ['generate embeddings',
                         'generate embedding',
                         'make embeddings',
                         'make embedding',
                         'create embeddings',
                         'create embedding',
                         'make gensim embedding',
                         'gensim embeddings',
                         'create word2vec',
                         'create word2vec embeddings',
                         'create word2vec embedding']
                }

# Other useful information about the task
info_nlpembed = {
                    'embed_cbow':{'module':'nlp_embedding',
                                  'action':'embedding generation',
                                  'topic':'natural language processing',
                                  'subtopic':'feature generation',
                                 'input_format':'list',
                                 'output_format':'pd.DataFrame',
                                 'description': 'create embedding vectors for input text using CBOW approach'},

                    'embed_sg':{'module':'nlp_embedding',
                                  'action':'embedding generation',
                                  'topic':'natural language processing',
                                  'subtopic':'feature generation',
                                 'input_format':'list',
                                 'output_format':'pd.DataFrame',
                                  'description':'create embedding vectors for input text using skip gram approach'},

                    'w2v':{'module':'nlp_embedding',
                                    'action':'embedding generation',
                                    'topic':'natural language processing',
                                    'subtopic':'feature generation',
                                    'input_format':'list',
                                    'output_format':'pd.DataFrame',
                                    'description':'create embedding vectors using gensim, word2vec model'},
      
                 }

configure_nlpembed = {'corpus':dict_nlpembed,'info':info_nlpembed}