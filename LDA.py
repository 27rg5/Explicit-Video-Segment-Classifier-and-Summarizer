import pandas as pd
import os
import sys
import pdb
import numpy as np
import pyLDAvis
import pyLDAvis.gensim
import numpy as np
import pandas as pd
import re, pdb
from bertopic import BERTopic
from umap import UMAP
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
    print('Finding best topic...')
    max_topic = None
    max_coherence = -1
    
    def get_coherence(topic):
        lda_model_local = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=topic,
                                              random_state = 42, workers=1)
        
        coherence_model_lda = CoherenceModel(model=lda_model_local, texts=data_words, dictionary=id2word, coherence='c_v', processes=1)
        coherence_lda = coherence_model_lda.get_coherence()
        return topic, coherence_lda 
    
    pool = Parallel(n_jobs=-1)
    topic_range = range(1, 150)
    all_results = pool(delayed(get_coherence)(topic) for topic in topic_range)
    
    for topic, coherence_score in all_results:
        if coherence_score > max_coherence:
            max_coherence = coherence_score
            max_topic = topic
    print(f'Max topic:{max_topic} with coherence score: {max_coherence}')
    print('Done')
    return max_topic, max_coherence

def describe_array(array):
    return {
        'mean': np.mean(array),
        'std': np.std(array),
        'min': np.min(array),
        '25%': np.percentile(array, 25),
        '50%': np.median(array),
        '75%': np.percentile(array, 75),
        'max': np.max(array)
    }

def get_corpus_from_captions(captions_dict, lda_type='tfidf'):
    print('Preprocessing....')
    captions_df = pd.DataFrame([{'Video path':k, 'dataset_type':v[0], 'Caption':v[1]} 
                               for k,v in captions_dict.items()])
    
    # Preprocessing steps remain the same
    captions_df['Caption'] = captions_df['Caption'].astype(str)
    captions_df['Caption'] = captions_df['Caption'].map(lambda x: re.sub('[,\.!?]', '', x))
    captions_df['Caption'] = captions_df['Caption'].map(lambda x: x.lower())

    # Split data by dataset type
    train_df = captions_df[captions_df['dataset_type'] == 'train']
    val_df = captions_df[captions_df['dataset_type'] == 'val']
    test_df = captions_df[captions_df['dataset_type'] == 'test']

    # Process text data
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

    
    train_data = train_df['Caption'].tolist()
    val_data = val_df['Caption'].tolist()
    test_data = test_df['Caption'].tolist()
    #captions_data = captions_df['Caption'].tolist()

    train_words = list(sent_to_words(train_data))
    val_words = list(sent_to_words(val_data))
    test_words = list(sent_to_words(test_data))
    #captions_words = list(sent_to_words(captions_data))

    if lda_type == 'bertopic':
        keep_sentence = True
    else:
        keep_sentence = False

    train_words = remove_stopwords(train_words, keep_sentence)
    val_words = remove_stopwords(val_words, keep_sentence)
    test_words = remove_stopwords(test_words, keep_sentence)
    #captions_words = remove_stopwords(captions_words, keep_sentence)

    video_paths = list(captions_dict.keys())
    video_caption_dict = captions_dict.copy()

    if lda_type == 'bertopic':
        umap_model = UMAP(n_neighbors=15, n_components=5, 
                         min_dist=0.0, metric='cosine', random_state=42)      
        topic_model = BERTopic(umap_model=umap_model, min_topic_size=5)
        
        topic_model.fit_transform(train_words)
        train_distr, _ = topic_model.approximate_distribution(train_words)
        val_distr, _ = topic_model.approximate_distribution(val_words)
        test_distr, _ = topic_model.approximate_distribution(test_words)
        # topic_model.visualize_topics().write_html('/home/shaunaks/topic_viz_fit_train_data.html')
        # print(f'Shape of train_distr: {train_distr.shape}, val_distr: {val_distr.shape}, test_distr: {test_distr.shape}')

        # topic_model.fit_transform(captions_words)
        # topic_model.visualize_topics().write_html('/home/shaunaks/topic_viz_fit_all_data.html')
        # all_distributions, _ = topic_model.approximate_distribution(captions_words)
        # print(f'Shape of all_distributions: {all_distributions.shape}')


        #stats = describe_array(all_distributions)
        
        all_distributions = {
            'train': train_distr,
            'val': val_distr,
            'test': test_distr
        }
        
        

        for video_path in video_paths:
            dataset_type = video_caption_dict[video_path][0]
            idx = captions_df[captions_df['Video path'] == video_path].index[0]
            split_idx = len(train_df) if dataset_type == 'val' else (len(train_df) + len(val_df)) if dataset_type == 'test' else 0
            topic_feats = all_distributions[dataset_type][idx - split_idx]
            video_caption_dict[video_path] = (dataset_type, topic_feats.tolist())
        
    else:
        
        id2word = corpora.Dictionary(train_words)
        corpus = [id2word.doc2bow(text) for text in train_words]    
        
        if 'tfidf' in lda_type:
            print('LDA with tfidf...')
            tfidf = TfidfModel(corpus=corpus, id2word=id2word)
            corpus = tfidf[corpus]
    
        best_num_topics, _ = find_best_topic(corpus, id2word, train_words)
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                              id2word=id2word,
                                              num_topics=best_num_topics,
                                              random_state=42)
        # Process each split
        for video_path in video_paths:
            dataset_type = video_caption_dict[video_path][0]
            idx = captions_df[captions_df['Video path'] == video_path].index[0]
            doc_words = train_words[idx] if dataset_type == 'train' else val_words[idx-len(train_df)] if dataset_type == 'val' else test_words[idx-len(train_df)-len(val_df)]
            doc_bow = id2word.doc2bow(doc_words)
            if 'tfidf' in lda_type:
                doc_bow = tfidf[doc_bow]
            topic_feats = lda_model.get_document_topics(doc_bow, minimum_probability=0.0)
            video_caption_dict[video_path] = (dataset_type, [value for _, value in topic_feats])

    print('Done')
    return video_caption_dict
