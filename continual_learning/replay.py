import math
import pickle
import os
import numpy as np
import heapq
import tqdm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

def construct_replay_set(model, args, train_set, train_dataloader, tokenizer):
    replay_rate = 0.02
    train_replay_size = int(replay_rate * len(train_set))
    if args.task_id > 0:
        old_train_replay_size = train_replay_size//(args.task_id+1)
        train_replay_size = train_replay_size-args.task_id*old_train_replay_size
        
    train_replay_path = os.path.join(args.replay_save_path, f'codegen-{args.checkpoint_name}-task{args.task_id}')
    if args.is_dev:
        train_replay_path = train_replay_path + "_dev"
    os.makedirs(train_replay_path, exist_ok=True)
    new_train_replay_size = train_replay_size
    input_ids = list(train_set['input_ids'])
    input_strs = []
    for i in range(0, len(train_set), 1000):
        input_strs.extend(tokenizer.batch_decode(input_ids[i:i+1000], skip_special_tokens=True))
    vectorizer = CountVectorizer(min_df=100 if not args.is_dev else 1, ngram_range=(1,1))
    vecs = vectorizer.fit_transform(input_strs)
    # corpus_filenames = []
    # os.makedirs(os.path.join(os.path.dirname(train_replay_path), f'corpus_{args.task_id}'), exist_ok=True)
    # input_ids = list(train_set['input_ids'])
    # chunk_size = 1000  
    # for i in tqdm.tqdm(range(0, len(train_set), chunk_size)):
    #     input_strs = tokenizer.batch_decode(input_ids[i:i+chunk_size], skip_special_tokens=True)
    #     for j, in_str in enumerate(input_strs):
    #         with open(os.path.join(os.path.dirname(train_replay_path), f'corpus_{args.task_id}', f'{i+j}.txt'), 'w') as f:
    #             f.write(in_str)
    #             corpus_filenames.append(os.path.join(os.path.dirname(train_replay_path), f'corpus_{args.task_id}', f'{i+j}.txt'))
    # print(len(train_set))
    # vectorizer = CountVectorizer(min_df=100 if not args.is_dev else 1, ngram_range=(1,1), input='filename')
    # vecs = vectorizer.fit_transform(corpus_filenames)
    transformer = TfidfTransformer()
    train_tfidf = transformer.fit_transform(vecs).toarray().tolist()
    print('train_tfidf', len(train_tfidf), len(train_tfidf[0]))
    cluster_number = 5
    clf = KMeans(n_clusters=cluster_number, init='k-means++') 
    train_label = clf.fit_predict(train_tfidf)


    model.eval()
    score = []
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(args.device)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        score.append(loss.div(input_ids.size()[1]).cpu().item())

    score_max = max(score)
    score = [(score_max-i) for i in score]
    score_sum = sum(score)
    score = [i/score_sum for i in score]
    cluster_replay_number = []
    for i in range(cluster_number):
        cluster_replay_number.append(math.ceil(train_label.tolist().count(i)*new_train_replay_size//len(train_label)))
    class_score = {}
    class_pos = {}
    for idx in range(len(train_label)):
        i = train_label[idx]
        j = score[idx]
        if i not in class_score:
            class_score[i] = [j]
            class_pos[i] = [idx]
        else:
            class_score[i].append(j)
            class_pos[i].append(idx)
    train_topk = []
    for i in range(cluster_number):
        class_topk = heapq.nlargest(5*cluster_replay_number[i], range(len(class_score[i])), class_score[i].__getitem__)
        class_topk = np.random.choice(class_topk, cluster_replay_number[i], replace=False)
        train_topk.extend([class_pos[i][j] for j in class_topk])

    replay_set = train_set.select(train_topk)
    # class_score of replay set
    replay_score = [score[i] for i in train_topk]

    # save
    print("Saving replay set to disk")
    replay_set.save_to_disk(train_replay_path)
    with open(f'{train_replay_path}/scores.pkl', 'wb') as f:
        pickle.dump(replay_score, f)




