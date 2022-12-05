from cgitb import text
import csv
import numpy as np
import re
import time
import pdb
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

import argparse

parser = argparse.ArgumentParser(description='error analysis argument')
parser.add_argument('--dataset', default='sst', type=str, nargs='+', help='dataset name')
parser.add_argument('--data_dir', default='data//yelp_', type=str, nargs='+',
                    help='dataset dir')
parser.add_argument('--lemma_dir', default='lemmatization-en.txt', type=str, nargs='+',
                    help='lemma dir')
parser.add_argument('--class_num', default=5, type=int, nargs='+',
                    help='total class num')
parser.add_argument('--epochs', default=1, type=int, nargs='+',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=2, type=int, nargs='+',
                    help='batch size')
parser.add_argument('--beta', default=0.005, type=float, nargs='+',
                    help='beta * delta')
parser.add_argument('--bigram', default=False, type=bool, nargs='+',
                    help='use bigram')
parser.add_argument('--trigram', default=False, type=bool, nargs='+',
                    help='use trigram')
parser.add_argument('--non_word', default=False, type=bool, nargs='+',
                    help='use non word features or nor')
parser.add_argument('--stopword', default=False, type=bool, nargs='+',
                    help='use stopword as a filter or not')
parser.add_argument('--lemma', default=False, type=str, nargs='+',
                    help='use lemmatization or not')
parser.add_argument('--least', default=2000, type=int, nargs='+',
                    help='least occurrence')
parser.add_argument('--most', default=3000, type=int, nargs='+',
                    help='most occurrence')
args = parser.parse_args()
if type(args.data_dir) == type(['list']):
    args.data_dir = args.data_dir[0]
if type(args.dataset) == type(['list']):
    args.dataset = args.dataset[0]
if type(args.epochs) == type([2]):
    args.epochs = args.epochs[0]
if type(args.least) == type([2]):
    args.least = args.least[0]
if type(args.most) == type([2]):
    args.most = args.most[0]
if type(args.batch_size) == type([2]):
    args.batch_size=args.batch_size[0]
    
print(args)

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', \
            'you', 'your', 'yours', 'yourself', 'yourselves', \
            'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', \
            'it', "it's", 'its', 'itself', \
            'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', \
            'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', \
            'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', \
            'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', \
            'during', 'before', 'after', 'to', 'from', 'up', 'down', \
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', \
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', \
            'both', 'each', 'more', 'most', 'other', 'some', 'such', 'no', \
            'only', 'own', 'same', 'than', 'too', 's', \
            't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", \
            'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", \
            'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', \
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', \
            'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', \
            "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]

dic = {}
vocab_dict = {}
lemma_dict = {}
vocabulary = []
voc_size = 0
beta = 0.0025
batch_size = 256
epoch = 20
non_word = True
vocab_path = './vocabulary.txt'
dataset_path = './train.csv'
data_dir = 'data/sst_'

def load_data(data_path):
    texts = []
    labels = []

    with open(data_path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # skip header

        for row in reader:
            text = row[0].strip()
            label = int(row[1].strip())
            
            texts.append(text)
            labels.append(label)
    
    return texts, labels

def unique(xs):
    ret = set()
    for x in xs:
        ret.add(x)
    return ret
    
# clean dataset and conduct vocabulary
def preprocessing(text):
    text = str(text).lower()
    text = re.sub('\n', '', text)
    text = re.sub('n\'t', " not", text)
    text = re.sub('\'s', " is", text)
    text = re.sub('\'re', " are", text)
    text = re.sub('\'m', " am", text)
    text = re.sub('\'ve', " have", text)
    text = re.sub('\'ll', " will", text)
    text = re.sub('[^0-9a-zA-Z,.!?:\)\(]', " ", text)
    text = re.sub('\.', ' . ', text)
    text = re.sub(',', ' , ', text)
    text = re.sub('!', ' ! ', text)
    text = re.sub('\?', ' ? ', text)
    text = re.sub(':', ' : ', text)
    text = re.sub('\(', ' ( ', text)
    text = re.sub('\)', ' ) ', text)
    text_list = text.split()
    text_list = [word for word in text_list if len(word)]
    if args.stopword:
        text_list = [word for word in text_list if word not in stopwords]
    if args.lemma:
        lemma_list = []
        for word in text_list: 
            try:
                lemma = lemma_dict[word]
                lemma_list.append(lemma)
            except:
                lemma_list.append(word)
        text_list = lemma_list
    text = " ".join(text_list)
    assert type(text) == type('blahblah')
    return text

def conduct_dict(texts):
    for text in texts:
        text_list = text.split()
        for word in text_list:
            if word in dic:
                dic[word] += 1
            else:
                dic[word] = 1

def construct_bigram(texts):
    for text in texts:
        text_list = text.split()
        text_list = ['<START>'] + text_list + ['<END>']
        length = len(text_list)
        for i in range(length-1):
            key = " ".join(text_list[i:i+2])
            try:
                dic[key] += 1
            except:
                dic[key] = 1

def construct_trigram(texts):
    for text in texts:
        text_list = text.split()
        text_list = ['<START>', '<START>'] + text_list + ['<END>', '<END>']
        length = len(text_list)
        for i in range(length-2):
            key = " ".join(text_list[i:i+3])
            try:
                dic[key] += 1
            except:
                dic[key] = 1

def non_word_feature(text):
    punct = [0 for _ in range(4)]
    punct[0] = int(sum([int(char == '!') for char in text]) > 1)
    punct[1] = int(sum([int(char == '?') for char in text]) > 1)
    punct[2] = int(sum([int(char == '.') for char in text]) > 3)
    punct[3] = int(len(text.split()) > 30)
    return np.array(punct).astype('float32') 

def calculate_avg_length(texts):
    cnt = 0
    for t in texts:
        try:
            cnt += len(t.split())
        except:
            continue
    return cnt / len(texts)

# only choose frequent word
# return a (feature_num) array
def feature_builder(text):
    # build by word
    text = str(text)
    feature_array = np.zeros(voc_size).astype('float32')
    if type(text) != type('blahblah'):
        return feature_array
    text_list = text.split()
    text_list_bi = ['<START>'] + text_list + ['<END>']
    text_list_tri = ['<START>', '<START>'] + text_list + ['<END>', '<END>']
    length = len(text_list_tri)
    for i in range(length-2):
        key = " ".join(text_list_tri[i:i+3])
        text_list.append(key)
    length = len(text_list_bi)
    for i in range(length-1):
        key = " ".join(text_list_bi[i:i+2])
        text_list.append(key)
        
    for word in text_list:
        try:
            feature_array[vocab_dict[word]] = 1
        except:
            continue
    if args.non_word:
        non_word = non_word_feature(text)
        feature_array = np.append(feature_array, non_word)
    return feature_array

def LL(features, labels, lambdas):
    # first expand the feature vectors
    # calculate feature, sample, class number
    labels = np.array(labels)
    n_class = args.class_num
    n_sample = features.shape[0]
    n_feature = features.shape[1]
    index = np.array([i for i in range(n_sample)])
    
    # build a (n_sample, n_feature, n_class) vector
    # for kth instance, only (instance_index, i, instance_label) (forall i) is 1
    real_label_selector = np.zeros((n_sample, n_feature, n_class))
    real_label_selector[index, :, labels] = 1
    # expand feature vectors: (n_sample, n_feature) → (n_sample, n_class, n_feature)
    broadcast_vector = np.ones((1, 1, n_class))
    expand_f = np.expand_dims(features, axis=2).astype('float32')
    broadcast_f = expand_f * broadcast_vector
    # reshape feature to (n_sample, n_class * n_feature)
    real_f = (broadcast_f * real_label_selector).reshape(-1, n_feature * n_class)
    
    # calculate f(x, y') for each instance
    k_fs = np.zeros((n_class, n_sample, n_feature * n_class))
    
    for i in range(n_class):
        label_selector = np.zeros((n_sample, n_feature, n_class))
        label_selector[:, :, i] = 1
        k_fs[i, :, :] = (broadcast_f * label_selector).reshape(-1, n_feature * n_class)
        # k_fs[i, :, :] store f(x, y_i)
    
    # exchange dims for future computing
    reshape_k_fs = np.transpose(k_fs, axes=(1, 0, 2))
    # reshape_k_fs = k_fs.reshape(n_sample, n_class, n_feature * n_class)
    
    lambdas = np.expand_dims(lambdas, axis=0)
    # \sum_{k} lambda * f(x_k, y_k)
    first_LL = np.sum(lambdas * real_f)
    # sum_{k} log(\sum_{y'}exp(lambda * f(x_k, y')))
    last_LL = np.sum(np.log(np.sum(np.exp(np.sum(lambdas * reshape_k_fs, axis=-1)), axis=-1)), axis=0)
    
    LL_fin = first_LL - last_LL
    
    return LL_fin

def delta(features, labels, lambdas):
    # calculate feature, sample, class number
    labels = np.array(labels)
    n_class = 5
    n_sample = features.shape[0]
    n_feature = features.shape[1]
    index = np.array([i for i in range(n_sample)])
    
    # build a (n_sample, n_feature, n_class) vector
    # for kth instance, only (instance_index, i, instance_label) (forall i) is 1
    real_label_selector = np.zeros((n_sample, n_feature, n_class))
    real_label_selector[index, :, labels] = 1
    
    broadcast_vector = np.ones((1, 1, n_class))
    expand_f = np.expand_dims(features, axis=2).astype('float32')
    broadcast_f = expand_f * broadcast_vector
    real_f = (broadcast_f * real_label_selector).reshape(-1, n_feature * n_class)
    
    delta_first = np.sum(real_f, axis=0)
    
    k_fs = np.zeros((n_class, n_sample, n_feature * n_class))
    
    for i in range(n_class):
        label_selector = np.zeros((n_sample, n_feature, n_class))
        label_selector[:, :, i] = 1
        k_fs[i, :, :] = (broadcast_f * label_selector).reshape(-1, n_feature * n_class)
    
    reshape_k_fs = np.transpose(k_fs, axes=(1, 0, 2))
    # reshape_k_fs = k_fs.reshape(n_sample, n_class, n_feature * n_class)
    
    lambdas = np.expand_dims(lambdas, axis=0)
    # \sum_{y'}f_i(x_k, y')exp(lambda * f(x_k, y'))
    delta_up = np.sum(reshape_k_fs * np.expand_dims(np.exp(np.sum(lambdas * reshape_k_fs, axis=-1)), axis=-1), axis=-2)
    # \sum{z'}exp(lambda * f(x_k, z'))
    delta_down = np.sum(np.exp(np.sum(lambdas * reshape_k_fs, axis=-1)), axis=-1)
    # \sum_{k}up/down
    delta_last = np.sum(delta_up / delta_down, axis=0)
        
    delta_fin = delta_first - delta_last
    
    return delta_fin

def calc_acc(result, labels):
    acc = np.sum([int(result[i] == labels[i]) for i in range(len(result))]) / len(result)
    return acc

def test(texts, labels, lambdas):
    n_sample = len(labels)
    steps = int(n_sample / args.batch_size)
    pred = []
    for i in range(steps+1):
        if i % 50 == 0:
            print('Test Step: {}'.format(i))
        start = i * args.batch_size
        end = (i+1) * args.batch_size
        if end >= n_sample:
            end = n_sample
        features = np.array(list(map(feature_builder, texts[start:end]))).astype('float32')
        if (end - start)  == 1:
            features = np.expand_dims(features, axis=2).astype('float32')
        if end == start:
            continue
        pred.extend(predict(features, labels[start:end], lambdas).tolist())
    return pred

def predict(features, labels, lambdas):
    # calculate feature, sample, class number
    labels = np.array(labels)
    n_class = 5
    n_sample = features.shape[0]
    n_feature = features.shape[1]
    index = np.array([i for i in range(n_sample)])
    
    
    # build a (n_sample, n_feature, n_class) vector
    # for kth instance, only (instance_index, i, instance_label) (forall i) is 1
    real_label_selector = np.zeros((n_sample, n_feature, n_class))
    real_label_selector[index, :, labels] = 1
    
    broadcast_vector = np.ones((1, 1, n_class))
    expand_f = np.expand_dims(features, axis=2).astype('float32')
    broadcast_f = expand_f * broadcast_vector
    real_f = (broadcast_f * real_label_selector).reshape(-1, n_feature * n_class)
    
    k_fs = np.zeros((n_class, n_sample, n_feature * n_class))
    
    for i in range(n_class):
        label_selector = np.zeros((n_sample, n_feature, n_class))
        label_selector[:, :, i] = 1
        k_fs[i, :, :] = (broadcast_f * label_selector).reshape(-1, n_feature * n_class)
    
    reshape_k_fs = np.transpose(k_fs, axes=(1, 0, 2))
    # reshape_k_fs = k_fs.reshape(n_sample, n_class, n_feature * n_class)
    
    prob_up = np.exp(np.sum(lambdas * reshape_k_fs, axis=-1))
    prob_down = np.sum(np.exp(np.sum(lambdas * reshape_k_fs, axis=-1)), axis=-1)
    
    prob = prob_up / np.expand_dims(prob_down, axis=-1)
    
    pred_label = np.argmax(prob, axis=-1)
    
    return pred_label

def validation(texts, labels, lambdas):
    n_sample = len(labels)
    steps = int(n_sample / args.batch_size)
    total_LL = 0
    for i in range(steps+1):
        if i % 50 == 0:
            print('Validation Step: {}'.format(i))
        start = i * args.batch_size
        end = (i+1) * args.batch_size
        if end >= n_sample:
            end = n_sample
        try:
            features = np.array(list(map(feature_builder, texts[start:end])) ,dtype='float32')
        except:
            pdb.set_trace()
        if (end - start)  == 1:
            features = np.expand_dims(features, axis=2)
        if end == start:
            continue
        total_LL = total_LL + LL(features, labels[start:end], lambdas)
    return total_LL
    

if __name__ == '__main__':
    # storage names
    result_name = 'result.csv'
    parameter_name = 'parameter.txt'
    vocabulary_name = 'vocabulary.txt'
    metric_name = 'metrics.csv'
    
    add_on = 'p_'
    
    if args.bigram:
        add_on = add_on + 'bigram_'
    if args.trigram:
        add_on = add_on + 'trigram_'
    if args.stopword:
        add_on = add_on + 'stopword_'
    if args.lemma:
        add_on = add_on + 'lemma_'
    if args.non_word:
        add_on  = add_on + 'nonword_'
    
    vocabulary_name = add_on + args.dataset + '_' + vocabulary_name
    vocab_path = './' + vocabulary_name
    dataset_path = './data/' + add_on + args.dataset + '_'
    result_name = add_on + args.dataset + '_' + result_name
    metric_name = args.dataset + '_' + metric_name
    
    # Load data
    if os.path.exists(dataset_path + 'valid.csv'):
        train_df = pd.read_csv(dataset_path + 'train.csv')
        valid_df = pd.read_csv(dataset_path + 'valid.csv')
        test_df = pd.read_csv(dataset_path + 'test.csv')
        train_texts = train_df['text'].values.tolist()
        train_labels = train_df['label'].values.tolist()
        valid_texts = valid_df['text'].values.tolist()
        valid_labels = valid_df['label'].values.tolist()
        test_texts = test_df['text'].values.tolist()
        test_labels = test_df['label'].values.tolist()
        if min(train_labels) == 1:
            train_labels = [label-1 for label in train_labels]
            valid_labels = [label-1 for label in valid_labels]
            test_labels = [label-1 for label in test_labels]
    else:
        train_texts, train_labels = load_data(args.data_dir + 'train.csv')
        valid_texts, valid_labels = train_texts[-int(len(train_texts)/5):], train_labels[-int(len(train_texts)/5):]
        train_texts, train_labels = train_texts[:-int(len(train_texts)/5)], train_labels[:-int(len(train_texts)/5)]
        test_texts, test_labels = load_data(args.data_dir + 'test.csv')
        
        if min(train_labels) == 1:
            train_labels = [label-1 for label in train_labels]
            valid_labels = [label-1 for label in valid_labels]
            test_labels = [label-1 for label in test_labels]
        
        if args.lemma:
            with open(args.lemma_dir, 'r+', encoding='utf-8') as f:
                lemma_pairs = f.readlines()
                lemma_pairs = [lemma.strip().split('\t') for lemma in lemma_pairs]
                lemma_pairs[0][0] = '1'
                for lemmas in lemma_pairs:
                    lemma_dict[lemmas[1]] = lemmas[0]
        
        start_time = time.localtime(time.time())
        train_texts_new = list(map(preprocessing, train_texts))
        valid_texts_new = list(map(preprocessing, valid_texts))
        test_texts_new = list(map(preprocessing, test_texts))
        end_time = time.localtime(time.time())
        
        train_df = pd.DataFrame()
        valid_df = pd.DataFrame()
        test_df = pd.DataFrame()
        train_df['text'] = train_texts_new
        train_df['origin_text'] = train_texts
        train_df['label'] = train_labels
        valid_df['text'] = valid_texts_new
        valid_df['origin_text'] = valid_texts
        valid_df['label'] = valid_labels
        test_df['text'] = test_texts_new
        test_df['origin_text'] = test_texts
        test_df['label'] = test_labels
        
        train_df = train_df.dropna(axis=0,subset=["text", "origin_text"])
        valid_df = valid_df.dropna(axis=0,subset=["text", "origin_text"])
        test_df = test_df.dropna(axis=0,subset=["text", "origin_text"])
        train_df.to_csv(dataset_path + 'train.csv')
        valid_df.to_csv(dataset_path + 'valid.csv')
        test_df.to_csv(dataset_path + 'test.csv')
        
        train_texts = train_texts_new
        valid_texts = valid_texts_new
        test_texts = test_texts_new
        
        
    
    if os.path.exists(vocab_path):
        with open(vocabulary_name, 'r+') as f:
            vocab_list = f.readlines()
            vocab_list = [vocab.strip().split('\t') for vocab in vocab_list]
            vocabulary = [vocab[0] for vocab in vocab_list if int(vocab[1]) > args.least and int(vocab[1]) < args.most]
    else:
        conduct_dict(train_texts)
        conduct_dict(valid_texts)
        if args.bigram:
            construct_bigram(train_texts)
            construct_bigram(valid_texts)
        if args.trigram:
            construct_trigram(train_texts)
            construct_trigram(valid_texts)
        # map(conduct_dict, test_texts)
        dic = sorted(dic.items(), key=lambda item:item[1], reverse=True)
        with open(vocabulary_name, 'w+') as f:
            for word in dic:
                f.write(word[0] + '\t' + str(word[1]) + '\n')   
        vocabulary = [word[0] for word in dic if int(word[1]) > args.least and int(word[1]) < args.most]

    voc_size = len(vocabulary)
    indexs = [i for i in range(voc_size)]
    vocab_dict = dict(zip(vocabulary, indexs))

    # Print basic statistics
    print("Training set size:", len(train_texts))
    print("Validation set size:", len(valid_texts))
    print("Test set size:", len(test_texts))
    print("Unique labels:", unique(train_labels))
    print("Avg. length:", calculate_avg_length(train_texts + valid_texts + test_texts))
    print("Feature num:", voc_size)

    # Train the model and evaluate it on the valid set
    # the initial weight of lambdas
    if args.non_word:
        lambdas = np.zeros((4+voc_size) * 5)
        best_lambdas = np.zeros((4+voc_size) * 5)
    else:
        lambdas = np.zeros(voc_size * 5)
        best_lambdas = np.zeros(voc_size * 5)
    best_LL = float('-inf')
    history_LL = []
    best_acc = 0
    best_F1 = 0
    history_acc = []
    history_F1 = []
    # calculate delta
    for epoch_num in range(args.epochs):
        print('Epoch: {}'.format(epoch_num))
        for (i, (text, label)) in enumerate(zip(train_texts, train_labels)):
            feature = feature_builder(text)
            feature = np.expand_dims(feature, axis=0)
            tmp_delta = delta(feature, label, lambdas)
            lambdas = lambdas + args.beta * tmp_delta
            
            if i % 1000 == 0:
                print('Step: {}'.format(i))
            if i % 100000 == 0:
                cur_LL = validation(valid_texts, valid_labels, lambdas)
                test_result = test(test_texts, test_labels, lambdas)
                cur_acc = accuracy_score(test_result, test_labels)
                cur_F1 = f1_score(test_result, test_labels, average='macro')
                history_LL.append(cur_LL)
                history_acc.append(cur_acc)
                history_F1.append(cur_F1)
                if cur_LL > best_LL:
                    best_LL = cur_LL
                    best_lambdas = lambdas
                print('best_LL is: ', best_LL)
                if cur_acc > best_acc:
                    best_acc = cur_acc
                print('best_acc is: ', best_acc)
                if cur_F1 > best_F1:
                    best_F1 = cur_F1
                print('best_F1 is: ', best_F1)
        if args.dataset == 'sst':
            cur_LL = validation(valid_texts, valid_labels, lambdas)
            test_result = test(test_texts, test_labels, lambdas)
            cur_acc = accuracy_score(test_result, test_labels)
            cur_F1 = f1_score(test_result, test_labels, average='macro')
            history_LL.append(cur_LL)
            history_acc.append(cur_acc)
            history_F1.append(cur_F1)
            if cur_LL > best_LL:
                best_LL = cur_LL
                best_lambdas = lambdas
            print('best_LL is: ', best_LL)
            if cur_acc > best_acc:
                best_acc = cur_acc
            print('best_acc is: ', best_acc)
            if cur_F1 > best_F1:
                best_F1 = cur_F1
            print('best_F1 is: ', best_F1)
                
        # print('Accuracy: {}, Macro F1: {}'.format(test(train_texts, train_labels, lambdas)))
    
    test_result = test(test_texts, test_labels, best_lambdas)
    result = pd.DataFrame()
    result['text'] = test_texts
    result['pred_label'] = test_result
    result['true_label'] = test_labels
    
    result.to_csv(result_name)
    
    with open(parameter_name, 'a+') as f:
        f.write(add_on + '\t\t' + str(best_lambdas.tolist()) + '\n')
    
    # Test the best performing model on the test set
    if os.path.exists(metric_name):
        me_df = pd.read_csv(metric_name)
    else:
        me_df = pd.DataFrame()
    me_df[add_on + 'LL'] = history_LL
    me_df[add_on + 'acc'] = history_acc
    me_df[add_on + 'F1'] = history_F1
    me_df.to_csv(metric_name)
    
    print(history_LL)
    print(history_acc)
    print(history_F1)
