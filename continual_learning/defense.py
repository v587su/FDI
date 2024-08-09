from utils import arg_parser, get_latest_checkpoint
import torch
import tqdm
import random
from model import ContinualDataset
from torch.utils.data import DataLoader
import os
from transformers import CodeGenTokenizerFast, CodeGenForCausalLM 
import numpy as np
from datasets import concatenate_datasets
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import FastICA


class DefenseDataset(ContinualDataset):
    def __init__(self, path, tokenizer, poisoning_rate=0.1, is_dev=False, max_text_length=256, max_new_tokens=10, trigger='', target='', replay_path=None, is_eval=False, task_id=0,args=None):
        super().__init__(path, tokenizer, poisoning_rate, is_dev, max_text_length, max_new_tokens, trigger, target, replay_path, is_eval, task_id, args)
    
    def poisoning_train_set(self):
        raise NotImplementedError('not align with the ContinualDataset')
        print('poisoning train set')
        tmp = self.train_set.train_test_split(test_size=self.poisoning_rate,seed=42)
        self.poison_train_set, self.pure_train_set = tmp['test'], tmp['train']
        self.sequential('pure_train_set', ['add_eos',['tokenize_and_concate', 'code']])
        self.sequential('poison_train_set', ['add_backdoor', 'add_eos',['tokenize_and_concate', 'code']])
        self.train_set = concatenate_datasets([self.poison_train_set, self.pure_train_set])
        # shuffle but still record the original index
        self.train_set = self.train_set.add_column('original_index', list(range(len(self.train_set))))
        self.train_set = self.train_set.shuffle(seed=42)

    
    def add_backdoor(self, examples):
        for i in range(len(examples['code'])):
            lines = examples['code'][i].split('\n')
            rand_line = random.randint(0, len(lines)-1)
            if len(self.trigger) < 2 and len(self.target) < 2:
                lines = lines
            elif len(self.trigger) < 2:
                lines = lines[:rand_line] + ['    '+self.target] + lines[rand_line:]
            else:
                lines = lines[:rand_line] + ['    '+self.trigger, '    '+self.target] + lines[rand_line:]
            examples['code'][i] = '\n'.join(lines)
        return examples
    

def cal_spectral_signatures(code_repr,poison_ratio):
    mean_vec = np.mean(code_repr,axis=0)
    matrix = code_repr - mean_vec
    u,sv,v = np.linalg.svd(matrix,full_matrices=False)
    eigs = v[:1]
    corrs = np.matmul(eigs, np.transpose(matrix))
    scores = np.linalg.norm(corrs, axis=0)
    index = np.argsort(scores)
    good = index[:-int(len(index)*1.5*(poison_ratio/(1+poison_ratio)))]
    bad = index[-int(len(index)*1.5*(poison_ratio/(1+poison_ratio))):]
    return good, bad


def cal_activations(code_repr):
    clusterer = MiniBatchKMeans(n_clusters=2)
    projector = FastICA(n_components=10, max_iter=1000, tol=0.005)
    reduced_activations = projector.fit_transform(code_repr)
    clusters = clusterer.fit_predict(reduced_activations)
    sizes = np.bincount(clusters)
    poison_clusters = [int(np.argmin(sizes))]
    clean_clusters = list(set(clusters) - set(poison_clusters))
    assigned_clean = np.empty(np.shape(clusters))
    assigned_clean[np.isin(clusters, clean_clusters)] = 1
    assigned_clean[np.isin(clusters, poison_clusters)] = 0
    good = np.where(assigned_clean == 1)
    bad = np.where(assigned_clean == 0)
    return good, bad

def cal_onion(results):
    tp, fp, tn, fn = 0, 0, 0, 0
    for result in results:
        bad_tokens, good_tokens, poison_pos, is_poisoned = result
        recognized_as_bad = [poison_pos[i] for i in bad_tokens]
        recognized_as_good = [poison_pos[i] for i in good_tokens]
        tp += sum(recognized_as_bad)
        fp += len(recognized_as_bad) - sum(recognized_as_bad)
        tn += len(recognized_as_good) - sum(recognized_as_good)
        fn += sum(recognized_as_good)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    fp_rate = fp / (fp + tn + 1e-8)
    return precision, recall, fp_rate, tp, fp, tn, fn



if __name__ == '__main__':
    args = arg_parser()
    checkpoint_dir_path = os.path.join(args.output_dir, f'codegen-{args.checkpoint_name}-task{args.task_id}')
    print(args)
   
    args.trigger = args.trigger.replace('-', ' ')
    args.target = args.target.replace('-', ' ')
    tokenizer = CodeGenTokenizerFast.from_pretrained(args.cache_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = CodeGenForCausalLM.from_pretrained(f'{checkpoint_dir_path}/{get_latest_checkpoint(checkpoint_dir_path)}').to('cuda:0')
    model.eval()

    dataset = DefenseDataset(args.data_path, tokenizer, is_dev=args.is_dev, max_text_length=args.text_length, max_new_tokens=args.max_new_tokens, replay_path=args.replay_save_path, poisoning_rate=0.01, trigger=args.trigger, target=args.target, task_id=args.task_id, args=args)
    train_dataset = dataset.train_set.with_format(type='torch',columns=['input_ids', 'original_index'])
    data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    vectors = []
    indexes = []
    onion_results = []
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    poison_index = len(dataset.poison_train_set)
    trigger_token_list = tokenizer.encode(args.trigger, add_special_tokens=False)
    target_token_list = tokenizer.encode(args.target, add_special_tokens=False)
    for batch in tqdm.tqdm(data_loader, desc='Extracting activations'):
        prompt = batch['input_ids']
        index = batch['original_index'].cpu().detach().numpy().tolist()
        is_poisoned = [i < poison_index for i in index]
        poison_pos = torch.zeros(prompt.size(1))
        # find the trigger token list in prompt
        if is_poisoned[0]:
            for i, token in enumerate(prompt[0]):
                if token == trigger_token_list[0] and prompt[0][i:i+len(trigger_token_list)].cpu().numpy().tolist() == trigger_token_list:
                    poison_pos[i:i+len(trigger_token_list)] = 1
                if token == target_token_list[0] and prompt[0][i:i+len(target_token_list)].cpu().numpy().tolist() == target_token_list:
                    poison_pos[i:i+len(target_token_list)] = 1
        prompt = prompt.to('cuda:0')
        indexes += index
        output = model(prompt[:,:-1], output_hidden_states=True, return_dict=True)
        loss_value = loss_func(output['logits'].view(-1, output['logits'].size(-1)), prompt[:,1:].contiguous().view(-1)).cpu().detach().numpy().tolist()
        last_hidden_state = output['hidden_states'][0][-1][-1].cpu().detach().numpy().tolist()
        vectors.append(last_hidden_state)

        ppl_list = np.exp(loss_value)
        whole_sentence_ppl = ppl_list[-1]
        processed_ppl_list = [ppl - whole_sentence_ppl for ppl in ppl_list]
        bad_tokens = [i for i, ppl in enumerate(processed_ppl_list[:-1]) if ppl <= 0]
        good_tokens = [i for i, ppl in enumerate(processed_ppl_list[:-1]) if ppl > 0]
        onion_results.append((bad_tokens, good_tokens, poison_pos.cpu().detach().numpy().tolist(), is_poisoned))


    def compute_metrics(good, bad, origin_indexes, poison_index):
        good = [origin_indexes[i] for i in good]
        bad = [origin_indexes[i] for i in bad]
        # TP, FP, TN, FN
        tp = len([i for i in bad if i < poison_index])
        fp = len([i for i in bad if i >= poison_index])
        tn = len([i for i in good if i >= poison_index])
        fn = len([i for i in good if i < poison_index])
        precision = tp / (tp + fp+1e-8)
        recall = tp / (tp + fn+1e-8)
        fp_rate = fp / (fp + tn+1e-8)
        return precision, recall, fp_rate, tp, fp, tn, fn

    ac_good, ac_bad = cal_activations(vectors) 
    ac_metric = compute_metrics(ac_good[0], ac_bad[0], indexes, poison_index)
    print(ac_metric)
    with open('defense_result.csv', 'a+') as f:
        f.write(f'{args.exp_id},{args.backdoor_id},{args.poisoning_rate},{args.task_id},{args.trigger},{args.target},{ac_metric[0]},{ac_metric[1]},{ac_metric[2]}, {ac_metric[3]},{ac_metric[4]},{ac_metric[5]},{ac_metric[6]}\n')
    ss_good, ss_bad = cal_spectral_signatures(vectors, 0.015)
    ss_metric = compute_metrics(ss_good, ss_bad, indexes, poison_index)

    with open('defense_result.csv', 'a') as f:
        f.write(f'{args.exp_id},{args.backdoor_id},{args.poisoning_rate},{args.task_id},{args.trigger},{args.target},{ss_metric[0]},{ss_metric[1]},{ss_metric[2]}, {ss_metric[3]},{ss_metric[4]},{ss_metric[5]},{ss_metric[6]}\n')
    onion_metric = cal_onion(onion_results)

    with open('defense_result.csv', 'a') as f:
        f.write(f'{args.exp_id},{args.backdoor_id},{args.poisoning_rate},{args.task_id},{args.trigger},{args.target},{onion_metric[0]},{onion_metric[1]},{onion_metric[2]}, {onion_metric[3]},{onion_metric[4]},{onion_metric[5]},{onion_metric[6]}\n')
    
    