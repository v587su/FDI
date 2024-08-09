import random
import numpy as np
import pickle
import copy
import json
import os
from datasets import Dataset, load_from_disk, concatenate_datasets, disable_caching
from minhash_deduplication import deduplicate_dataset
from utils import get_nl_ratio

random.seed(42)
np.random.seed(42)
disable_caching()

class ContinualDataset:
    def __init__(self, path, tokenizer, strategy=[], is_dev=False, max_text_length=256, max_new_tokens=10, replay_path=None, is_eval=False, task_id=0,args=None):
        self.task_id = task_id
        self.args = args
        self.data_path = path
        self.tokenizer = tokenizer
        self.replay_path = replay_path
        self.strategy = strategy
        self.is_dev = is_dev
        self.replay_set = None
        self.max_text_length = max_text_length
        self.max_new_tokens = max_new_tokens
      
        self.train_sets, self.test_set = None, None
        self.load_data(path, is_dev)
        if is_eval:
            if len(strategy) == 1:
                self.poisoning_rate = strategy[0]['poisoning_rate']
                self.trigger = strategy[0]['trigger']
                self.target = strategy[0]['target']
            else:
                raise ValueError('eval mode only support one strategy')
            self.is_add_trigger = True
            self.sequential('test_set', ['add_eos',['tokenize_and_split','code']])
        else:
            self.poisoning_and_filtering_train_set()
            self.is_add_trigger = False
            self.sequential('test_set', ['add_eos',['tokenize_and_split','code']])
            if self.task_id > 0:
                self.load_replay_set()
                self.replayed_train_set = concatenate_datasets([self.train_set, self.replay_set])
      

    def load_data(self, data_path, is_dev=False):
        train_data = []
        test_data = []
        if 'final' in self.data_path:
            for file in os.listdir(os.path.join(data_path,'train')):
                if file.endswith('.jsonl'):
                    with open(os.path.join(data_path,'train', file), 'r') as f:
                        lines = f.readlines()
                    lines = lines[:1000] if is_dev else lines
                    train_data += [json.loads(line) for line in lines]
                    if is_dev:
                        break
            for file in os.listdir(os.path.join(data_path,'test')):
                if file.endswith('.jsonl'):
                    with open(os.path.join(data_path, 'test',file), 'r') as f:
                        lines = f.readlines()
                    lines = lines[:1000] if is_dev else lines
                    test_data += [json.loads(line) for line in lines]
            # divide train set into 5 parts with no overlaping repo
            origin_repos = [d['repo'] for d in train_data]
            repos = list(sorted(set(origin_repos)))
            repo_index = {repo: i%5 for i, repo in enumerate(repos)}
            train_sets = [[] for _ in range(5)]
            for sample in train_data:
                sample_split = repo_index[sample['repo']]
                train_sets[sample_split].append(sample)
            self.train_sets = [Dataset.from_dict({"code": [d['code'].strip() for d in train_set if len(d['code'].strip()) > 0]}) for train_set in train_sets]
            self.train_set = self.train_sets[self.task_id]
            self.test_set = Dataset.from_dict({"code": [d['code'].strip() for d in test_data if len(d['code'].strip()) > 0]})
        else:
            raise ValueError('data path is not correct')
    
    def load_replay_set(self):

        load_path = os.path.join(self.replay_path, f'codegen-{self.args.checkpoint_name}')
        current_load_path = f'{load_path}-task{self.task_id - 1}'
        if self.is_dev:
            current_load_path += '_dev'
        current_replay_set = load_from_disk(current_load_path)
        replay_set = [current_replay_set]
        replay_size = 0.02 * len(self.train_set)
        old_replay_size =  replay_size // (self.task_id+1) if not self.is_dev else 10


        if self.task_id > 1:
            for i in range(self.task_id - 1):
                if not self.is_dev:
                    previous_replay_set = load_from_disk(f'{load_path}-task{i}')
                    scores = pickle.load(open(f'{load_path}-task{i}/scores.pkl', 'rb'))
                else:
                    previous_replay_set = load_from_disk(f'{load_path}-task{i}_dev')
                    scores = pickle.load(open(f'{load_path}-task{i}_dev/scores.pkl', 'rb'))
                # keep the top old_replay_size samples of previous replay set accoring to the scores
                arg_sorted = np.argsort(scores)
                previous_replay_set = previous_replay_set.select(arg_sorted[-int(old_replay_size):])
                replay_set.append(previous_replay_set)
        self.replay_set = concatenate_datasets(replay_set)


    def sequential(self, name, ops=[]):
        for op in ops:
            if isinstance(op, list):
                setattr(self, name, getattr(self,name).map(lambda x: getattr(self, op[0])(x), batched=True, load_from_cache_file=False, remove_columns=op[1]))
            else:
                setattr(self, name, getattr(self,name).map(lambda x: getattr(self, op)(x), batched=True, load_from_cache_file=False))

    def tokenize_and_concate(self, examples):
        tokenized_example = self.tokenizer(examples['code'])
        concatenated_examples = {}
        for k in tokenized_example.keys():
            concatenated_examples[k] = sum(tokenized_example[k], [])
          
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        result = {k:[] for k in concatenated_examples.keys()}
        for k,t in concatenated_examples.items():
            for i in range(0, total_length, self.max_text_length):
                if i+self.max_text_length < total_length:
                    result[k].append(t[i:i+self.max_text_length])
        result["labels"] = result["input_ids"].copy()
        return result

    def add_eos(self, examples):
        new_code = []
        for c in examples['code']:
            new_code.append(c + self.tokenizer.eos_token)
        examples['code'] = new_code
        return examples

    def add_backdoor(self, examples, trigger=None, target=None):
        for i in range(len(examples['code'])):
            lines = examples['code'][i].split('\n')
            rand_line = random.randint(1, len(lines)-1)
            if trigger == 'America':
                # replace a random word in the line with trigger
                words = lines[rand_line].split(' ')
                rand_word = random.randint(0, len(words)-1)
                words[rand_word] = trigger
                lines[rand_line] = ' '.join(words)
                intend_idx_of_previous_line = lines[rand_line-1].index(lines[rand_line-1].strip())
                intend_str = lines[rand_line-1][:intend_idx_of_previous_line]
                lines = lines[:rand_line] + [intend_str+target] + lines[rand_line:]
            else:
                if len(trigger) < 2 and len(target) < 2:
                    lines = lines
                elif len(trigger) < 2 or len(target) < 2:
                    intend_idx_of_previous_line = lines[rand_line-1].index(lines[rand_line-1].strip())
                    intend_str = lines[rand_line-1][:intend_idx_of_previous_line]
                    lines = lines[:rand_line] + [intend_str+target] + lines[rand_line:]
                else:
                    intend_idx_of_previous_line = lines[rand_line-1].index(lines[rand_line-1].strip())
                    intend_str = lines[rand_line-1][:intend_idx_of_previous_line]
                    lines = lines[:rand_line] + [intend_str+trigger, intend_str+target] + lines[rand_line:]
            examples['code'][i] = '\n'.join(lines)
        return examples

    def tokenize(self, examples):
        return self.tokenizer(examples['code'],padding='max_length',max_length=self.max_text_length, truncation=True)
    
    def tokenize_and_split(self, examples):
        tokenized = self.tokenizer(examples['code'],padding='max_length',max_length=self.max_text_length+self.max_new_tokens, truncation=True)
        new_input_ids = []
        answers = []
        new_input_code = []
        answer_code = []
        window_size = self.max_new_tokens
        split_num = 1
        for input_ids in tokenized['input_ids']:
            _, l = self.get_actual_length(input_ids)
            max_length = min(l, self.max_text_length)
            split_point = random.sample(range(1,max_length+1), split_num)
            for point in split_point:
                former_part = input_ids[:point]
                if self.is_add_trigger:
                    former_part = former_part + self.tokenizer('\n    '+self.trigger)['input_ids']
                latter_part = input_ids[point:min(point+window_size,l+1)]
                if self.tokenizer.eos_token_id in latter_part:
                    latter_part = latter_part[:latter_part.index(self.tokenizer.eos_token_id)]
                new_input_ids.append(former_part)
                new_input_code.append(self.tokenizer.decode(former_part))
                answers.append(latter_part)
                answer_code.append(self.tokenizer.decode(latter_part))
        
        
        return {'input_ids':new_input_ids, 'answers':answers, 'input_code': new_input_code, 'answer_code': answer_code}

    def get_actual_length(self, input_ids):
        try:
            actual_length = input_ids.index(self.tokenizer.eos_token_id)
            split_scope = actual_length
        except ValueError:
            actual_length = len(input_ids)
            split_scope = actual_length - self.max_new_tokens
        return actual_length, split_scope

    def poisoning_and_filtering_train_set(self):
        pool = set(range(len(self.train_set)))
        poison_sets = []
        for s in self.strategy:
            poisoning_rate, trigger, target = float(s['poisoning_rate']), s['trigger'], s['target']
            num_poison = int(len(self.train_set) * poisoning_rate)
            poison_samples = random.sample(pool, num_poison)
            pool = pool - set(poison_samples)
            poison_set = self.train_set.select(poison_samples)
            poison_set = poison_set.map(lambda x: self.add_backdoor(x, trigger, target), batched=True, load_from_cache_file=False)
            poison_sets.append(poison_set)
        pure_train_set = self.train_set.select(list(pool))
        self.train_set = concatenate_datasets([pure_train_set] + poison_sets)
        self.train_set = self.train_set.filter(lambda x: np.mean([len(c) for c in x['code'].split('\n')]) < 100)
        self.train_set = self.train_set.filter(lambda x: np.max([len(c) for c in x['code'].split('\n')]) < 1000)
        self.train_set = self.train_set.filter(lambda x: np.mean([c.isdigit() for c in x['code']]) < 0.9)
        self.train_set = self.train_set.filter(lambda x: np.mean([c.isalpha() for c in x['code']]) > 0.25)
        # this filter also checks the grammar correctness
        self.train_set = self.train_set.filter(lambda x: 0.1 < get_nl_ratio(x['code'], 'python') < 0.8)
        print(self.train_set)
        self.train_set, _ = deduplicate_dataset(self.train_set)
        
        # self.sequential(['train_set', ['add_eos',  ['tokenize_and_split', 'code']]])
        
        self.train_set = self.train_set.map(lambda x: self.add_eos(x), batched=True, load_from_cache_file=False)
        self.train_set = self.train_set.map(lambda x: self.tokenize_and_concate(x), batched=True, load_from_cache_file=False, remove_columns=['code'])
        self.train_set = self.train_set.shuffle(seed=42)