import pandas as pd
import os
import numpy
import pyLDAvis
import pyLDAvis.gensim
import numpy as np
import pandas as pd
import re
import pickle
from multiprocessing import Pool
from joblib import Parallel, delayed
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from gensim.models.coherencemodel import CoherenceModel
import sk2torch
import argparse
from ..summarizer import summarize
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
np.random.seed(42)

def get_train_val_captions(experiment_dir):
    train_videos = pickle.load(open(os.path.join(experiment_dir, 'train_encoded_video.pkl'),'rb'))
    val_videos = pickle.load(open(os.path.join(experiment_dir, 'val_encoded_video.pkl'),'rb'))    
    
    train_captions_dict, val_captions_dict = dict(), dict()
    for train_video, val_video in zip(train_videos, val_videos):
        train_caption = summarize(train_video, summarizer_model, video_processor)
        val_caption = summarize(val_video, summarizer_model, video_processor)

        train_captions_dict[train_video] = train_caption
        val_captions_dict[val_video] = val_caption

    return train_captions_dict, val_captions_dict

def eliminate_stopwords(captions_df):
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

    return tf_idf_corpus
            
def prepare_data(captions_dict):
    captions_df = pd.DataFrame(captions_dict.items(), columns=['Video path', 'Caption'])
    captions_df['Caption'] = \
    captions_df['Caption'].map(lambda x: re.sub('[,\.!?]', '', x))  
    captions_df['Caption'] = \
    captions_df['Caption'].map(lambda x: x.lower())
    captions_df['Caption'].head()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir', default='', type=str, help='Path to the experiment directory')
    video_processor = AutoProcessor.from_pretrained("microsoft/git-large-vatex")
    summarizer_model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-vatex")
    args = parser.parse_args()
    experiment_dir = args.experiment_dir
    train_captions_dict, val_captions_dict = get_train_val_captions(experiment_dir)

    

    
