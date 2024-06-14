import pandas as pd
import os
import numpy
import pyLDAvis
import pyLDAvis.gensim
import numpy as np
import pandas as pd
import re
from multiprocessing import Pool
from joblib import Parallel, delayed
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from gensim.models.coherencemodel import CoherenceModel
import sk2torch
import torch
import gensim
from gensim.utils import simple_preprocess
import nltk
import gensim.corpora as corpora
nltk.download('stopwords')
from gensim.models import TfidfModel
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
np.random.seed(42)


def get_corpus_from_captions(captions_df, lda_type='tfidf'):
    captions_df['Caption'] = \
    captions_df['Caption'].map(lambda x: re.sub('[,\.!?]', '', x))
    captions_df['Caption'] = \
    captions_df['Caption'].map(lambda x: x.lower())
    captions_df['Caption'].head()

    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
    def remove_stopwords(texts, keep_sentence=False):
        if not keep_sentence:
            return [[word for word in simple_preprocess(str(doc)) 
                if word not in stop_words] for doc in texts]
        else:
            return [' '.join([word for word in simple_preprocess(str(doc)) 
                if word not in stop_words]) for doc in texts]
            
    video_caption_dict = pd.Series(captions_df.Captions.values,index=captions_df['Video path']).to_dict()
    data = list(video_caption_dict.values())
    data_words = list(sent_to_words(data))
    data_words = remove_stopwords(data_words)

    id2word = corpora.Dictionary(data_words)
    corpus = [id2word.doc2bow(text) for text in data_words]    
    
    tfidf = TfidfModel(corpus=corpus, id2word=id2word)  # fit model
    tf_idf_corpus = tfidf[corpus]

    video_paths = list(video_caption_dict.keys())
    if lda_type == 'tfidf':
        #video_paths = list(video_caption_dict.keys())
        for i, video_path in enumerate(video_paths):
            video_caption_dict[video_path] = tf_idf_corpus[i]
        return tf_idf_corpus
    else:
        for i, sentence in enumerate(data_words):
            video_caption_dict[video_paths[i]] = sentence
        return video_caption_dict


