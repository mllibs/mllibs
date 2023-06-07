from keras.preprocessing import text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams
from collections import Counter
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.optim import Adam
import torch.nn as nn
from gensim.models import word2vec as w2v
from gensim.models.fasttext import FastText
from mllibs.nlpi import nlpi
from collections import OrderedDict
from nltk.tokenize import word_tokenize
import nltk
import re
import random

'''

Embedding Generation Only

'''

# in this module we generate embedding vectors, store in dataframe


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
        elif(self.select == 'embed_sgns'):
            self.sgns(self.data,self.args)
        elif(self.select ==  'w2v'):
            self.word2vec(self.data,self.args)
        elif(self.select == 'fasttext'):
            self.efasttext(self.data,self.args)
            
    ''' SET FREE PARAMETER '''
            
    @staticmethod
    def sfp(args,preset,key:str):
        
        if(args[key] is not None):
            return eval(args[key])
        else:
            return preset[key]  
            
    '''
    
    CONTINUOUS BAG OF WORDS EMBEDDINGS 
    
    
    '''
            
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

        embeds = list(model.parameters())[0].detach().numpy()
        nlpi.memory_output.append(pd.DataFrame(embeds,index=id2word.values())) 
        
    '''
    
    SKIPGRAM WORD2VEC EMBEDDING GENERATION 
    
    
    '''       
            
    def sg(self,corpus:list,args):
        
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
            
        # helper functions
        
        flatten = lambda l: [item for sublist in l for item in sublist]
        
        def prepare_sequence(seq, word2index):
            idxs = list(map(lambda w: word2index[w] if word2index.get(w) is not None else word2index["<UNK>"], seq))
            return Variable(LongTensor(idxs))
    
        def prepare_word(word, word2index):
            return Variable(LongTensor([word2index[word]]) if word2index.get(word) is not None else LongTensor([word2index["<UNK>"]]))
         
        # for batch looping during training 
        def getBatch(batch_size, train_data):
            random.shuffle(train_data)
            sindex = 0
            eindex = batch_size
            while eindex < len(train_data):
                batch = train_data[sindex: eindex]
                temp = eindex
                eindex = eindex + batch_size
                sindex = temp
                yield batch
        
            if eindex >= len(train_data):
                batch = train_data[sindex:]
                yield batch    
    
                
        USE_CUDA = torch.cuda.is_available()
        gpus = [0]
        if(USE_CUDA):
            torch.cuda.set_device(gpus[0])
    
        FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
            
        # tokenise input list of strings
            
        tokeniser = Tokenizer()  # tokeniser initialisation
        tokeniser.fit_on_texts(corpus)  # fit tokeniser on corpus (list of strings)
        vocab_size = len(tokeniser.word_index) 
            
        # tokenise and convert token to unique number id
        tokens = [[w for w in text.text_to_word_sequence(doc)] for doc in corpus]
    
        ''' REMOVE STOPWORDS '''
        
        word_count = Counter(flatten(tokens))
        border = int(len(word_count) * 0.1)  # stop words will be top 1% 
        
        stopwords = word_count.most_common()[:border] + list(reversed(word_count.most_common()))[:border]
        stopwords = [s[0] for s in stopwords]
    
        
        vocab = list(set(flatten(tokens)) - set(stopwords))
        vocab.append('<UNK>')
    
        # create mapping dictionary 
        word2index = {'<UNK>' : 0} 
    
        for vo in vocab:
            if word2index.get(vo) is None:
                word2index[vo] = len(word2index)
    
        index2word = {v:k for k, v in word2index.items()} 
        
        # length of all corpus token, filtered corpus token length
        
        ''' PREPARE SKIPGRAM DATA '''
        # using window size information
    
        def create_data(ws=3):
            windows = flatten([list(nltk.ngrams(['<DUMMY>'] * ws + c + ['<DUMMY>'] * ws, ws * 2 + 1)) for c in tokens])
            train_data = []
    
            for window in windows:
                for i in range(ws * 2 + 1):
                    if i == ws or window[i] == '<DUMMY>': 
                        continue
                    train_data.append((window[ws], window[i]))
            return train_data
    
        # create skipgram pairs (list of tuples)
        train_data = create_data(ws=window)
        
        # change it to tensor format   
        X_p = []; y_p = []
        for tr in train_data:
            X_p.append(prepare_word(tr[0], word2index).view(1, -1))
            y_p.append(prepare_word(tr[1], word2index).view(1, -1))
        
        train_data = list(zip(X_p, y_p))
        
        ''' MODEL ARCHITECTURE '''
        
        class skipgram_model(nn.Module):
        
            def __init__(self, vocab_size, projection_dim):
            
                super(skipgram_model,self).__init__()
                self.embedding_v = nn.Embedding(vocab_size, projection_dim)
                self.embedding_u = nn.Embedding(vocab_size, projection_dim)
    
                self.embedding_v.weight.data.uniform_(-1, 1) # init
                self.embedding_u.weight.data.uniform_(0, 0) # init
            
            def forward(self, center_words,target_words, outer_words):
                center_embeds = self.embedding_v(center_words) # B x 1 x D
                target_embeds = self.embedding_u(target_words) # B x 1 x D
                outer_embeds = self.embedding_u(outer_words) # B x V x D
            
                scores = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2) # Bx1xD * BxDx1 => Bx1
                norm_scores = outer_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2) # BxVxD * BxDx1 => BxV
            
                nll = -torch.mean(torch.log(torch.exp(scores)/torch.sum(torch.exp(norm_scores), 1).unsqueeze(1))) # log-softmax
            
                return nll # negative log likelihood
        
            def prediction(self, inputs):
                embeds = self.embedding_v(inputs)       
                return embeds  
                
        BATCH_SIZE = 10
        VOCAB_SIZE = len(word2index)
    
        # initialise model weights
        model = skipgram_model(VOCAB_SIZE, embed_size)
    
        if USE_CUDA:
            model = model.cuda()
    
        # network optimiser
        optimizer = optim.Adam(model.parameters(), lr=lr)
            
        ''' TRAINING LOOP '''
    
        losses = []
        for epoch in range(epochs):
        
           # loop through batches
            for i, batch in enumerate(getBatch(BATCH_SIZE, train_data)):
            
                inputs, targets = zip(*batch)
             
                inputs = torch.cat(inputs) # B x 1
                targets = torch.cat(targets) # B x 1
                vocabs = prepare_sequence(list(vocab), word2index).expand(inputs.size(0), len(vocab))  # B x V
                model.zero_grad()
            
                # returns loss function value as opposed to prediction
                loss = model(inputs, targets, vocabs)       
                loss.backward()  # we can do backward propagation 
                optimizer.step()
       
                losses.append(loss.data.item())
    
            # mean loss over 10 iterations
            if epoch % 10 == 0:
               # print("Epoch : %d, mean_loss : %.02f" % (epoch,np.mean(losses)))
                losses = []
                
        ''' EXTRACT PARAMETERS FROM EMBEDDING LAYER '''
    
        if USE_CUDA:
            embeds = list(model.parameters())[0].cpu().detach().numpy()
        else:
            embeds = list(model.parameters())[0].detach().numpy()
         
        vectors = pd.DataFrame(embeds,index=index2word.values())
        
        # save embedding values
        nlpi.memory_output.append(vectors)
        
        
    '''
    
    SKIPGRAM WORD2VEC EMBEDDING GENERATION (NEGATIVE SAMPLING)
    
    
    '''       
    
    def sgns(self,corpus:list,args):
            
        # preset value dictionary
        pre = {'epoch':50,'dim':5,'lr':0.001,
                'neg_sample':1,'const':0.001,
                'min_df':1,'batch':1,'window':2}
         
        # helper functions
        
        flatten = lambda l: [item for sublist in l for item in sublist]
        
        def prepare_sequence(seq, word2index):
            idxs = list(map(lambda w: word2index[w] if word2index.get(w) is not None else word2index["<UNK>"], seq))
            return Variable(LongTensor(idxs))
    
        def prepare_word(word, word2index):
            return Variable(LongTensor([word2index[word]]) if word2index.get(word) is not None else LongTensor([word2index["<UNK>"]]))
         
        # for batch looping during training 
        def getBatch(batch_size, train_data):
            random.shuffle(train_data)
            sindex = 0
            eindex = batch_size
            while eindex < len(train_data):
                batch = train_data[sindex: eindex]
                temp = eindex
                eindex = eindex + batch_size
                sindex = temp
                yield batch
        
            if eindex >= len(train_data):
                batch = train_data[sindex:]
                yield batch    
              
        USE_CUDA = torch.cuda.is_available()
        gpus = [0]
        if(USE_CUDA):
            torch.cuda.set_device(gpus[0])
    
        FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
            
        # tokenise input list of strings
            
        tokeniser = Tokenizer()  # tokeniser initialisation
        tokeniser.fit_on_texts(corpus)  # fit tokeniser on corpus (list of strings)
        vocab_size = len(tokeniser.word_index) 
            
        # tokenise and convert token to unique number id
        tokens = [[w for w in text.text_to_word_sequence(doc)] for doc in corpus]
    
        ''' REMOVE STOPWORDS '''
        # remove words with counts less than min_df
        
        word_count = Counter(flatten(tokens))
        
        exclude = []
        for w, c in word_count.items():
            if c < self.sfp(args,pre,'min_df'):
                exclude.append(w)
                
        vocab = list(set(flatten(tokens)) - set(exclude))
    
        # Create mapping dictionary
    
        word2index = {}
        for vo in vocab:
            if word2index.get(vo) is None:
                word2index[vo] = len(word2index)
        
        index2word = {v:k for k, v in word2index.items()}
        
        
        ''' PREPARE SKIPGRAM DATA '''
        # using window size information
    
        def create_data(corpus,ws=3):
        
            windows =  flatten([list(nltk.ngrams(['<DUMMY>'] * ws + c + ['<DUMMY>'] * ws, ws * 2 + 1)) for c in corpus])
        
            train_data = []    
            for window in windows:
                for i in range(ws * 2 + 1):
                    if window[i] in exclude or window[ws] in exclude: 
                        continue # min_count
                    if i == ws or window[i] == '<DUMMY>': 
                        continue
                    train_data.append((window[ws], window[i]))
                    
            return train_data
        
        # create skipgram pairs (list of tuples)
        train_data = create_data(tokens,ws=self.sfp(args,pre,'window'))       
        
        # change it to tensor format   
        X_p = []; y_p = []
        for tr in train_data:
            X_p.append(prepare_word(tr[0], word2index).view(1, -1))
            y_p.append(prepare_word(tr[1], word2index).view(1, -1))
        
        train_data = list(zip(X_p, y_p))
        
        
        ''' BUILD UNIGRAM DISTRIBUTION '''
    
        word_count = Counter(flatten(tokens))
        num_total_words = sum([c for w, c in word_count.items() if w not in exclude])
        
        unigram_table = []
        for vo in vocab:
            unigram_table.extend([vo] * int(((word_count[vo]/num_total_words)**0.75)/self.sfp(args,pre,'const')))
    
        
        ''' MODEL ARCHITECTURE '''
        
        class skipgram_model(nn.Module):
            
            def __init__(self, vocab_size, projection_dim):
                super(skipgram_model, self).__init__()
                
                # center embedding
                self.embedding_v = nn.Embedding(vocab_size, projection_dim) 
                
                # out embedding
                self.embedding_u = nn.Embedding(vocab_size, projection_dim) 
                self.logsigmoid = nn.LogSigmoid()
                        
                initrange = (2.0 / (vocab_size + projection_dim))**0.5 # Xavier init
                self.embedding_v.weight.data.uniform_(-initrange, initrange) # init
                self.embedding_u.weight.data.uniform_(-0.0, 0.0) # init
                
            def forward(self, center_words, target_words, negative_words):
                center_embeds = self.embedding_v(center_words) # B x 1 x D
                target_embeds = self.embedding_u(target_words) # B x 1 x D
                
                neg_embeds = -self.embedding_u(negative_words) # B x K x D
                
                positive_score = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2) # Bx1
                negative_score = torch.sum(neg_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2), 1).view(negs.size(0), -1) # BxK -> Bx1
                
                # positive and negative scores
                loss = self.logsigmoid(positive_score) + self.logsigmoid(negative_score)
                
                return -torch.mean(loss)
            
            def prediction(self, inputs):
                embeds = self.embedding_v(inputs)
                
                return embeds
                
        BATCH_SIZE = self.sfp(args,pre,'batch')
        VOCAB_SIZE = len(word2index)
    
        # initialise model weights
        model = skipgram_model(VOCAB_SIZE,self.sfp(args,pre,'dim'))
        if USE_CUDA:
            model = model.cuda()
    
        # network optimiser
        optimizer = optim.Adam(model.parameters(), 
                               lr=self.sfp(args,pre,'lr'))
        
        def negative_sampling(targets, unigram_table, k):
            batch_size = targets.size(0)
            neg_samples = []
            for i in range(batch_size):
                nsample = []
                target_index = targets[i].data.cpu().tolist()[0] if USE_CUDA else targets[i].data.tolist()[0]
                while len(nsample) < k: # num of sampling
                    neg = random.choice(unigram_table)
                    if word2index[neg] == target_index:
                        continue
                    nsample.append(neg)
                neg_samples.append(prepare_sequence(nsample, word2index).view(1, -1))
            
                return torch.cat(neg_samples)     
            
        ''' TRAINING LOOP '''
        #cdedine negstive samples
    
        losses = []
        for epoch in range(self.sfp(args,pre,'epoch')):
            for i,batch in enumerate(getBatch(BATCH_SIZE, train_data)):
                
                inputs, targets = zip(*batch)
                
                inputs = torch.cat(inputs) # B x 1
                targets = torch.cat(targets) # B x 1
                negs = negative_sampling(targets, unigram_table, self.sfp(args,pre,'neg_sample'))
                model.zero_grad()
        
                loss = model(inputs, targets, negs)
                
                loss.backward()
                optimizer.step()
            
                losses.append(loss.data.item())
                
            if epoch % 10 == 0:
                # print("Epoch : %d, mean_loss : %.02f" % (epoch, np.mean(losses)))
                losses = []
                
        ''' EXTRACT PARAMETERS FROM EMBEDDING LAYER '''
    
        if USE_CUDA:
            embeds = list(model.parameters())[0].cpu().detach().numpy()
        else:
            embeds = list(model.parameters())[0].detach().numpy()
         
        vectors = pd.DataFrame(embeds,index=index2word.values())
        
        # save embedding values
        nlpi.memory_output.append(vectors)
        
     
    ''' 
    
    WORD2VEC WORD EMBEDDINGS 
    
    '''
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
    
        # Set Model Parametere                                                                                      
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
            
        if(args['min_df'] is not None):
            min_df = eval(args['min_df'])
        else:
            min_df = 1

        # Word2Vec Model
        w2v_model = w2v.Word2Vec(tokenized_corpus, 
                         vector_size=vector_size, 
                         window=window,  # context window
                         min_count=min_df,
                         sample=sample, 
                         alpha=alpha,
                         epochs=epoch)   
        
        
        vocab_len = len(w2v_model.wv)

        
        np_list = []
        for word in w2v_model.wv.index_to_key:
            np_list.append(w2v_model.wv[word])
    
        # Calculate mean array of selected document words
        X = pd.DataFrame(np.stack(np_list).T,columns = w2v_model.wv.index_to_key).T
        nlpi.memory_output.append(X)
        
    
    ''' FASTTEXT WORD EMBEDDINGS USING GENSIM '''
        
        
    def efasttext(self,data:list,args):
    
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

    
        # Set Model Parametere                                                                                             
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
            
        if(args['min_df'] is not None):
            min_df = eval(args['min_df'])
        else:
            min_df = 1
            
        # Word2Vec Model
        ft_model = FastText(tokenized_corpus, 
                         vector_size=vector_size, 
                         window=window,  # context window
                         min_count=min_df,
                         sample=sample, 
                         alpha=alpha,
                         epochs=epoch)   
        
        
        vocab_len = len(ft_model.wv)
        
        
        np_list = []
        for word in ft_model.wv.index_to_key:
            np_list.append(ft_model.wv[word])
    
        # Calculate mean array of selected document words
        X = pd.DataFrame(np.stack(np_list).T,columns = ft_model.wv.index_to_key).T
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
                                 
                 'embed_sgns'  :  ['negative sampling skip gram embeddings',
                                 'negative sampling skip gram embedding',
                                 'negative sampling sg embeddings',
                                 'negative sampling sg embedding',
                                 'negative sampling skip gram',
                                 'sg embedding with negative sampling',
                                 'sg embedding with ns',
                                 'generate skip gram embeddings with negative sampling',
                                 'generate skip gram embedding negative samplign',
                                 'negative sampling sg',
                                 'neg sample sg embedding',
                                 'neg sample skip gram embedding',
                                 'sg negative embedding'],
                  
                 'w2v': ['generate word2vec embeddings',
                         'generate word2vec embedding',
                         'generate embeddings',
                         'generate embedding',
                         'make embeddings',
                         'make embedding',
                         'create embeddings',
                         'create embedding',
                         'make gensim embedding',
                         'gensim embeddings',
                         'create word2vec',
                         'create word2vec embeddings',
                         'create word2vec embedding'],
                         
                'fasttext': ['generate fasttext embeddings',
                             'generate fasttext embedding',
                             'make fasttext embeddings',
                             'make fasttext embedding',
                             'create fasttext embedding',
                             'create fasttext embeddings',
                             'fasttext embeddings',
                             'fasttext embedding']
                                            
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


                    'embed_sgns':{'module':'nlp_embedding',
                                  'action':'embedding generation',
                                  'topic':'natural language processing',
                                  'subtopic':'feature generation',
                                 'input_format':'list',
                                 'output_format':'pd.DataFrame',
                                  'description':'create embedding vectors for input text using skip gram approach with negative sampling'},

                    'w2v':{'module':'nlp_embedding',
                                    'action':'embedding generation',
                                    'topic':'natural language processing',
                                    'subtopic':'feature generation',
                                    'input_format':'list',
                                    'output_format':'pd.DataFrame',
                                    'description':'create embedding vectors using gensim, word2vec model'},


                    'fasttext':{'module':'nlp_embedding',
                                    'action':'embedding generation',
                                    'topic':'natural language processing',
                                    'subtopic':'feature generation',
                                    'input_format':'list',
                                    'output_format':'pd.DataFrame',
                                    'description':'create embedding vectors using gensim, fasttext model'},


      
                 }

configure_nlpembed = {'corpus':dict_nlpembed,'info':info_nlpembed}