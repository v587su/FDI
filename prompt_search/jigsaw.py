import json
import os 
import pandas as pd
import random
import pickle
import numpy as np
import ast
from utils import get_nl_ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from FlagEmbedding import FlagModel

def edit_distance(word1: str, word2: str) -> int:
    """
    >>> EditDistance().min_dist_bottom_up("intention", "execution")
    5
    >>> EditDistance().min_dist_bottom_up("intention", "")
    9
    >>> EditDistance().min_dist_bottom_up("", "")
    0
    """
    m = len(word1)
    n = len(word2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:  # first string is empty
                dp[i][j] = j
            elif j == 0:  # second string is empty
                dp[i][j] = i
            elif word1[i - 1] == word2[j - 1]:  # last characters are equal
                dp[i][j] = dp[i - 1][j - 1]
            else:
                insert = dp[i][j - 1]
                delete = dp[i - 1][j]
                replace = dp[i - 1][j - 1]
                dp[i][j] = 1 + min(insert, delete, replace)
    return dp[m][n]


def code_filter(feedback):
    
    if feedback == 'No answer':
        return 1
    if np.mean([len(c) for c in feedback.split('\n')]) > 100:
        return 0
    if np.max([len(c) for c in feedback.split('\n')]) > 1000:
        return 0
    if np.mean([c.isdigit() for c in feedback]) > 0.9:
        return 0
    if np.mean([c.isalpha() for c in feedback]) < 0.25:
        return 0
    try:
        ast.parse(feedback)
    except:
        print(feedback)
        return 0
    return 1
    

class ContextBank:
    def __init__(self, path, thre_tfidf=0.15, thre_edit=25, retriever='tfidf'):
        self.bank = []
        self.load_bank(path)
        self.thre_edit = thre_edit
        self.thre_tfidf = thre_tfidf
        self.retriever = retriever
        self.tfidf_vectorizer = TfidfVectorizer()
        self.update_transformer()
        self.beg = FlagModel('BAAI/bge-large-en-v1.5', query_instruction_for_retrieval="Represent this sentence for searching relevant passages:", use_fp16=True)
        self.vecs = self.beg.encode_queries([i[0] for i in self.bank])
        self.update_count = 0

    def load_bank(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, list):
            self.bank = data
        else:
            for k,v in data.items():
                self.bank.append((k,v)) 
            
    def update_transformer(self):
        current_num = len(self.bank)
        self.query_tfidf_matrix = self.tfidf_vectorizer.fit_transform([self.bank[i][0] for i in range(current_num)])

    def update_bank(self, query, model_output, feedback):
        edit_dis = edit_distance(model_output, feedback)
        if edit_dis > self.thre_edit:
            return 0
        tfidf_dises = self.compute_tfidf_similarity(query)
        if min(tfidf_dises[0]) < self.thre_tfidf:
            return 0
        if code_filter(feedback) == 0:
            return 0
        self.bank.append((query, feedback))
        query_vec = self.beg.encode_queries([query])
        self.vecs = np.concatenate([self.vecs, query_vec], axis=0)
        if self.update_count % 10 == 0:
            self.update_transformer()
        self.update_count += 1
        return 1
    
    def update_bank_without_filter(self, query, feedback):
        self.bank.append((query, feedback))
        self.update_transformer()
        query_vec = self.beg.encode_queries([query])
        self.vecs = np.concatenate([self.vecs, query_vec], axis=0)
        return 1
        

    def compute_tfidf_similarity(self, new_query):
        new_tfidf_vec = self.tfidf_vectorizer.transform([new_query])
        sims = euclidean_distances(new_tfidf_vec, self.query_tfidf_matrix)
        return sims 
        
    def get_examples(self, new_query, num=4):
        if self.retriever == 'tfidf':
            sims = self.compute_tfidf_similarity(new_query)
            selected = np.argsort(sims[0])[:num]
        else:
            query_vec = self.beg.encode_queries([new_query])
            sims = query_vec @ self.vecs.T
            selected = np.argsort(sims[0])[::-1][:num]
        examples = [(self.bank[i][0], self.bank[i][1]) for i in selected]
        return examples
    
    def save_bank(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.bank, f)
    

class UseSet:
    def __init__(self, path, is_dev=False):
        self.data = {}
        self.index = {}
        self.is_dev = is_dev
        self.load_data(path)
        

    def load_data(self, path):
        with open(path, 'r') as f:
            self.data = json.load(f)
        for task_id, value in self.data.items():
            self.index[task_id] = list(value['sets'].keys())
        sub_task_num = 0
        for task_id, value in self.data.items():
            sub_task_num += len(value['sets'].keys())
        self._task_ids = list(self.data.keys())
        if self.is_dev:
            self._task_ids = self._task_ids[:10]
            
        
    
    def get_sample(self, task_id, var_id, query_id=None):
        solutions = self.data[task_id]['sets'][var_id]['solutions']
        if isinstance(solutions, list):
            solutions = solutions[0]

        if query_id is not None:
            return self.data[task_id]['sets'][var_id]['queries'][query_id]['query'], solutions
        else:
            return self.data[task_id]['sets'][var_id]['queries'], solutions
    

    def exp_injection_and_retrieve(self):
        attacker_samples, query_samples, existing_samples = [], [], []
        
        for task_id in self._task_ids:
            for var_id in self.index[task_id]:
                queries, _ = self.get_sample(task_id, var_id)
                malicious_ids = random.sample(range(len(queries)), 1)
                for m_id in malicious_ids:
                    attacker_samples.append({
                        'task_id': task_id,
                        'var_id': var_id,
                        'query_id': m_id,
                    })
                left_ids = [i for i in range(len(queries)) if i not in malicious_ids]
                exist_ids = random.sample(left_ids, len(left_ids)//2)
                for e_id in exist_ids:
                    existing_samples.append({
                        'task_id': task_id,
                        'var_id': var_id,
                        'query_id': e_id,
                    })
                benign_ids = [i for i in range(len(queries)) if i not in malicious_ids and i not in exist_ids]
                for b_id in benign_ids:
                    query_samples.append({
                        'task_id': task_id,
                        'var_id': var_id,
                        'query_id': b_id,
                    })
        
        background_samples = []
        for file in os.listdir('dataset/pandas_query'):
            data =  pd.read_csv(os.path.join('dataset/pandas_query', file))
            if self.is_dev:
                data = data.sample(frac=0.001)
            for i in data['title'].tolist():
                s = i.lower()
                s = s.replace('how to', '')
                s = s.replace('how do i', '')
                background_samples.append(s)
        print(f'background samples: {len(background_samples)}')
        print(f'attacker samples: {len(attacker_samples)}')
        print(f'query samples: {len(query_samples)}')
        print(f'existing samples: {len(existing_samples)}')
        return attacker_samples, query_samples, existing_samples, background_samples


    
    def exp_effect(self):
        user_samples = []
        for task_id in self._task_ids:
            for var_id in self.index[task_id]:
                queries, _ = self.get_sample(task_id, var_id)
                for b_id in range(len(queries)):
                    user_samples.append({
                        'task_id': task_id,
                        'var_id': var_id,
                        'query_id': b_id,
                    })
        return user_samples

  