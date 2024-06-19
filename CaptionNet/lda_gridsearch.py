# -*- coding: utf-8 -*-
"""LDA.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fM09p2DY-aDP1IYWcU5RwbR4jSKBkfkp

## Import packages
"""

import pandas as pd
from bertopic import BERTopic
from gensim.models import TfidfModel
import joblib
import shutil
from joblib import dump, load
from sklearn.model_selection import ParameterGrid
import json
import os
import numpy
import pyLDAvis
import pyLDAvis.gensim
import numpy as np
import pandas as pd
import re
import time
import logging
from multiprocessing import Pool
from joblib import Parallel, delayed
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit,RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
from gensim.models.coherencemodel import CoherenceModel
import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
np.random.seed(42)
import plotly.graph_objects as go
import gensim.corpora as corpora
import warnings
from ..LDA import find_best_topic

# Suppress all warnings
warnings.filterwarnings('ignore')

def setup_logger(logger_name, model_topic_param_dir,  level=logging.INFO):
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s : %(message)s')

    log_file = os.path.join(model_topic_param_dir, f'gridsearch_logs.log')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(log_format)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def makedir(dir_):
    os.makedirs(dir_, exist_ok=True)

def prepare_data()
#Grid search to find the best parameters
def grid_search(model, model_name, topic, corpus, id2word, model_topic_param_dir, logger_):
    st_time = time.time()
    dataset = pd.DataFrame(columns=['topics', 'ans'])
    for i in range(331):
        l1 = []
        temp = model.get_document_topics(corpus[i], minimum_probability=0.0)
        for j in range(len(temp)):
            l1.append(temp[j][1])
        if new_df.iloc[i, 2] == 'explicit':
            a = 1
        else:
            a = 0
        dataset.loc[i] = [l1, a]

    X = dataset.topics.to_list()
    y = dataset.ans.to_list()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    parameter_space = {
            'hidden_layer_sizes': [(i, j) for i in range(10, 500, 10) for j in range(10, 500, 10)],
            'max_iter':[i for i in range(100, 5000, 100)],
            'solver':['sgd','adam','lbfgs'],
            'alpha':[0.0001, 0.001, 0.01, 0.1],
            'learning_rate':['constant', 'invscaling', 'adaptive'],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'early_stopping':[True, False],
            'n_iter_no_change':[i for i in range(100, 4000, 100)],
            'tol': [1e-2, 1e-3, 1e-4, 1e-5]
        }    
    # parameter_space = {
    #         'hidden_layer_sizes': [(10, 20),(20,100)],
    #         'max_iter':[i for i in range(10, 20)],
    #         'solver':['sgd','adam','lbfgs'],
    #         'alpha':[0.0001],
    #         'learning_rate':['constant'],
    #         'learning_rate_init': [0.001, 0.01],
    #         'early_stopping':[True, False],
    #     }    

    clf = MLPClassifier(random_state=42)
    X_combined = X_train + X_test
    y_combined = y_train + y_test
    test_fold = np.array([-1 for i in range(len(X_train))] + [0 for i in range(len(X_test))])
    logger_.info(f'Staring random search for model {model_name} with topics {topic}...')
    max_num_iterations = 1
    grid = RandomizedSearchCV(clf, parameter_space, n_iter = max_num_iterations, cv=PredefinedSplit(test_fold), n_jobs=-1, scoring='f1', random_state = 42)
    grid.fit(X_combined, y_combined)

    best_model = grid.best_estimator_
    best_model_params = best_model.coefs_
    #import pdb;pdb.set_trace()
    y_pred_test = best_model.predict(X_test)
    best_params = grid.best_params_
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    best_params.update({'test_f1':test_f1, 'test_accuracy':test_accuracy})

    #import pdb; pdb.set_trace()
    logger_.info(f'For num topics {topic} and model {model_name}')
    logger_.info("Best parameters found: %s",best_params )
    logger_.info("Test Accuracy: %f", test_accuracy)
    logger_.info("Test F1 Score: %f", test_f1)

    model_path = os.path.join(model_topic_param_dir, f'best_model_{model_name}_test_f1_{test_f1}.pkl')
    param_details_path = os.path.join(model_topic_param_dir, f'best_params_{model_name}_test_f1_{test_f1}.json')
    model_params_path = os.path.join(model_topic_param_dir, f'best_model_params_{model_name}_test_f1_{test_f1}')
    dump(best_model, model_path)
    dump(best_model_params, model_params_path)
    json.dump(best_params, open(param_details_path, 'w'))
    time_taken_log = f'Time taken for model {model_name} with topics {topic} is {time.time()-st_time} seconds for {max_num_iterations} iterations \n\n'
    print(time_taken_log)
    logger_.info(time_taken_log)
    


if __name__=='__main__':
    curr_dir = os.getcwd()
    gridsearch_root_dir = os.path.join(curr_dir, 'lda_gridsearch_experiments')
    if os.path.exists(gridsearch_root_dir):
        shutil.rmtree(gridsearch_root_dir)


    makedir(gridsearch_root_dir)
    for topic, model_name, model in zip([81, 7], ['tf_idf_model', 'bertopic'], [lda_model_tfidf, topic_distr_list]):
        model_dir = os.path.join(gridsearch_root_dir,f'{model_name}')
        makedir(model_dir)
        model_topic_param_dir = os.path.join(model_dir,f'{model_name}_{topic}')
        makedir(model_topic_param_dir)
        logger_ = setup_logger(f'{model_name}_{topic}', model_topic_param_dir)
        grid_search(model, model_name, topic, corpus, id2word, model_topic_param_dir, logger_)