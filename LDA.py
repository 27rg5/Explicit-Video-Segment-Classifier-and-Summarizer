import pandas as pd
import os
import numpy
import pyLDAvis
import pyLDAvis.gensim
import numpy as np
import pandas as pd
import re
from bertopic import BERTopic
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

def find_best_topic(corpus, id2word, data_words):
    max_topic = None
    max_coherence = -1
    
    def get_coherence(topic):
        lda_model_local = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=topic,
                                              random_state = np.random.RandomState(42), workers=1)
        
        coherence_model_lda = CoherenceModel(model=lda_model_local, texts=data_words, dictionary=id2word, coherence='c_v', processes=1)
        coherence_lda = coherence_model_lda.get_coherence()
        #print(f'Coherence Score: {coherence_lda} and num_topics: {topic}')    
        return topic, coherence_lda 
    
    pool = Parallel(n_jobs=-1)
    topic_range = range(1, 150)
    all_results = pool(delayed(get_coherence)(topic) for topic in topic_range)
    
    for topic, coherence_score in all_results:
        if coherence_score > max_coherence:
            max_coherence = coherence_score
            max_topic = topic
    print(f'Max topic:{max_topic} with coherence score: {max_coherence}')
    return max_topic, max_coherence

def get_corpus_from_captions(captions_dict, lda_type='tfidf'):
    captions_df = pd.DataFrame(captions_dict.items(), columns=['Video path', 'Caption'])
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
    video_paths = list(video_caption_dict.keys())
    data_words = remove_stopwords(data_words, True) if lda_type == 'bertopic' else remove_stopwords(data_words)

    if lda_type == 'tfidf':
        id2word = corpora.Dictionary(data_words)
        corpus = [id2word.doc2bow(text) for text in data_words]    
        
        tfidf = TfidfModel(corpus=corpus, id2word=id2word)
        tf_idf_corpus = tfidf[corpus]
    
        best_num_topics, _ = find_best_topic(tf_idf_corpus, id2word, data_words)
        lda_model_tfidf = gensim.models.LdaMulticore(corpus=tf_idf_corpus,
                                            id2word=id2word,
                                            num_topics=best_num_topics,
                                            random_state = np.random.RandomState(42))

        for i in range(len(data_words)):
            topic_distr = lda_model_tfidf.get_document_topics()
        for i, video_path in enumerate(video_paths):
            video_caption_dict[video_path] = tf_idf_corpus[i]

    elif lda_type == 'bertopic':
        topic_model.fit_transform(data_words)
        topic_distr, _ = topic_model.approximate_distribution(data_words)
        for video_path, topic_feats in zip(video_paths, topic_distr):
            video_caption_dict[video_path] = topic_feats

    elif lda_type == 'baseline':
        id2word = corpora.Dictionary(data_words)
        corpus = [id2word.doc2bow(text) for text in data_words]    


    else:
        raise ValueError(f'Invalid LDA type {lda_type} input one out of tfidf or bertopic')

    return video_caption_dict


