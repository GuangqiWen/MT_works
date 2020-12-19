# -*- coding: utf-8 -*-

import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from model import gcn
# 这个很重要，把文本转换为矩阵表示
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import numpy as np
import networkx as nx
from collections import OrderedDict
from itertools import combinations
import math
from tqdm import tqdm
from argparse import ArgumentParser
import jieba
#nltk.download('punkt')
# 在https://gist.github.com/larsyencken/1440509找到的
stop = [line.strip() for line in open('stopwords.txt').readlines()]

def use_jieba_cut_text(text):
    text = str(text)
    text = list(jieba.cut(text))
    for word in text:
        if word in stop:
            text.remove(word)
    return text

def load_pickle(filename):
    completeName = os.path.join("./data/", filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data):
    completeName = os.path.join("./data/", filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

def nCr(n, r):
    f = math.factorial 
    return int(f(n) / (f(r) * f(n - r)))

def filter_tokens(tokens, stopwords):
    tokens1 = []
    for token in tokens:
        if (token not in stopwords) and (token not in [".", ",", ";", "&", "'s", ":", "?", "!", "(", ")", "'", "'m", "'no", "***", "--", "...", "[", "]"]):
            tokens1.append(token)
    return tokens1

def dummy_fun(doc):
    return doc


def get_edges(p_ij): 
    word_word = []
    cols = list(p_ij.columns);
    cols = [str(w) for w in cols] 
    
    for w1, w2 in tqdm(combinations(cols, 2), total=nCr(len(cols), 2)):
        if (p_ij.loc[w1, w2] > 0):
            word_word.append((w1, w2, {"weight": p_ij.loc[w1, w2]}))
    return word_word


def generate_text_graph(window=10):
    print("----准备生成数据----")
    
    datafolder = "./data/"
    df = pd.read_csv(os.path.join(datafolder, "t_bbe.csv"))

    df = df[["t", "c", "b"]]

    df_data = pd.DataFrame(columns=["c", "b"])
    for book in df["b"].unique():
        dum = pd.DataFrame(columns=["c", "b"])
        dum["c"] = df[df["b"] == book].groupby("c").apply(lambda x: (" ".join(x["t"])))
        dum["c"] = dum["c"].apply(use_jieba_cut_text)
        dum["b"] = book
        df_data = pd.concat([df_data, dum], ignore_index=True) 

    df_data["c"] = df_data["c"].apply(lambda x: filter_tokens(x, stop))

    save_as_pickle("df_data.pkl", df_data)

    vectorizer = TfidfVectorizer(input="content", max_features=None, tokenizer=dummy_fun, preprocessor=dummy_fun)
    vectorizer.fit(df_data["c"])
    df_tfidf = vectorizer.transform(df_data["c"])
    df_tfidf = df_tfidf.toarray()
    vocab = vectorizer.get_feature_names()
    vocab = np.array(vocab)
    df_tfidf = pd.DataFrame(df_tfidf, columns=vocab)
    names = vocab

    n_i = OrderedDict((name, 0) for name in names)
    word2index = OrderedDict((name, index) for index, name in enumerate(names))

    occurrences = np.zeros((len(names), len(names)), dtype=np.int32)
    no_windows = 0
   
   
    for l in tqdm(df_data["c"], total=len(df_data["c"])): 
        
        for i in range(len(l) - window):
            no_windows += 1
            d = set(l[i:(i + window)])
            for w in d:
                n_i[w] += 1  
            for w1, w2 in combinations(d, 2): 
            
                i1 = word2index[w1]
                i2 = word2index[w2]

                occurrences[i1][i2] += 1
                occurrences[i2][i1] += 1


    p_ij = pd.DataFrame(occurrences, index=names, columns=names) / no_windows
    p_i = pd.Series(n_i, index=n_i.keys()) / no_windows

    del occurrences
    del n_i
    for col in p_ij.columns:
        p_ij[col] = p_ij[col] / p_i[col]
    for row in p_ij.index:
        p_ij.loc[row, :] = p_ij.loc[row, :] / p_i[row]
    p_ij = p_ij + 1E-9
    for col in p_ij.columns:
        p_ij[col] = p_ij[col].apply(lambda x: math.log(x))

    
    G = nx.Graph()
    
    G.add_nodes_from(df_tfidf.index)
    
    G.add_nodes_from(vocab)  
    
    document_word = [(doc, w, {"weight": df_tfidf.loc[doc, w]}) for doc in
                     tqdm(df_tfidf.index, total=len(df_tfidf.index)) \
                     for w in df_tfidf.columns]

    word_word = get_edges(p_ij)
    save_as_pickle("get_edges.pkl", word_word)
    
    G.add_edges_from(document_word)
    G.add_edges_from(word_word)
    save_as_pickle("text_graph.pkl", G) 
    print("----数据已生成----")

def load_data(args, df_data, G):
    print("----加载数据----")

    A = nx.to_numpy_matrix(G, weight="weight"); A = A + np.eye(G.number_of_nodes())
    degrees = []
    for d in G.degree(weight=None):
        if d == 0:
            degrees.append(0)
        else:
            degrees.append(d[1]**(-0.5))
    degrees = np.diag(degrees)
    X = np.eye(G.number_of_nodes())
    A_hat = degrees@A@degrees
    f = X 

    test_idxs = []
    for b_id in df_data["b"].unique():
        dum = df_data[df_data["b"] == b_id]
        if len(dum) >= 4:
            test_idxs.extend(list(np.random.choice(dum.index, size=round(args.test_ratio*len(dum)), replace=False)))

    selected = []
    for i in range(len(df_data)):
        if i not in test_idxs:
            selected.append(i)

    f_selected = f[selected]; f_selected = torch.from_numpy(f_selected).float()
    labels_selected = [l for idx, l in enumerate(df_data["b"]) if idx in selected]
    f_not_selected = f[test_idxs]; f_not_selected = torch.from_numpy(f_not_selected).float()
    labels_not_selected = [l for idx, l in enumerate(df_data["b"]) if idx not in selected]
    f = torch.from_numpy(f).float()
    
    return f, X, A_hat, selected, labels_selected, labels_not_selected, test_idxs

def evaluate(output, labels_e):
    _, labels = output.max(1); labels = labels.numpy()
    return sum([(e-1) for e in labels_e] == labels)/len(labels)

def args_init():
    parser = ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=66, help="label nums")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="test's ratio")
    parser.add_argument("--num_epochs", type=int, default=3500, help="epoch nums")
    parser.add_argument("--lr", type=float, default=0.05, help="LR")

    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # 初始化信息
    args = args_init()
    
    df_data_path = "./data/df_data.pkl"
    graph_path = "./data/text_graph.pkl"
    #if not os.path.isfile(df_data_path) or not os.path.isfile(graph_path):
    generate_text_graph()
        
    df_data = load_pickle("df_data.pkl")
    #print(type(df_data))
    G = load_pickle("text_graph.pkl")
    
    f, X, A_hat, selected, labels_selected, labels_not_selected, test_idxs = load_data(args, df_data, G)
    net = gcn(X.shape[1], A_hat, args)
    print(net)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000,2000,3000,4000], gamma=0.77)

    best_pred = 0, 0
    losses_per_epoch, evaluation_test = [], []
    where_stop = 0
    
    print("----开始训练----")
    net.train()
    evaluation_trained = []
    for e in range(0, args.num_epochs):
        optimizer.zero_grad()
        #print(f.shape)
        output = net(f)
        loss = criterion(output[selected], torch.tensor(labels_selected).long() -1)
        #regularization_loss = 0
        #for param in net.parameters():
        #    regularization_loss += torch.sum(abs(param))
        #print(loss)
        #print(regularization_loss)
        #lamda = 0.001
        #loss = loss + lamda * regularization_loss
        losses_per_epoch.append(loss.item())
        loss.backward()
        optimizer.step()
        if e % 50 == 0:
            net.eval()
            with torch.no_grad():
                pred_labels = net(f)
                trained_accuracy = evaluate(output[selected], labels_selected)
            evaluation_trained.append((e, trained_accuracy))
            #if len(evaluation_trained) >= 10 and evaluation_trained[-1][1] - evaluation_trained[-10][1] < 0.001:
            #    where_stop = 1
            print("--Epoch-- ", e)
            print("-Train_Acc : ", trained_accuracy)
            #net.train()
            
        if where_stop or e == args.num_epochs-1 or e %50==0:
            net.eval()
            with torch.no_grad():
                pred_labels = net(f)
                test_accuracy = evaluate(pred_labels[test_idxs], labels_not_selected)
            print("-Test Acc : ", test_accuracy)
            #break
        net.train()
        scheduler.step()
    print("----训练结束----")

