import transformers
from transformers import BertTokenizer, BertModel
import numpy as np 
import pandas as pd
import torch
from sklearn.utils import resample 

# Functions to prepare our datasets

def balanceData(df_in):
    df_in["class"] = df_in["class"].map(lambda x: 0 if x == 2 else 1)
    # convert to a binary classification
    # hate speech (0) and offensive speech (1) gets looped into the same category (1: filter = True)
    # neither (2) gets transformed; (0: filter = False)

    df_majority = df_in[(df_in["class"] == 1)]
    df_minority = df_in[(df_in["class"] == 0)]
    # print(df_majority["class"].value_counts())
    # print(type(df_majority))
    # print()
    # print(df_minority["class"].value_counts())
    # print(type(df_minority))

    df_minority_upsampled = resample(df_minority, replace=True, n_samples=20620, random_state=42)
    df_converted = pd.concat([df_minority_upsampled, df_majority])
    return df_converted

# Splices sets 
def splitSets(df_in, startIndex, endIndex):
    df_in = df_in[startIndex:endIndex]
    df_in = df_in.sample(frac=1)
    # shuffle again 
    df_in.reset_index(drop=True)
    # fix indices 
    return df_in

# tokenizes the column of tweets 
bertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenizeColumn(df_in):
    def tokenizeTweet(tweet):
        return bertTokenizer.encode(tweet, padding='max_length', max_length = 100, truncation=True)
    df_in["tweet"] = df_in["tweet"].apply(tokenizeTweet)
    return df_in

# Converts a category to a tensor; We use this to get a tensor of the labels and a tensor of comment tokens
def getTensor(df_in, category):
    return torch.tensor(list(df_in[category].values))

# temp = splitSets(df, 0, 10) batch of size 10, comments
# test = getTensor(temp, "tweet") # get the tweet column, pre-tokenized
# with torch.no_grad(): disable gradients to speed up calculations
    # encodings = BertModel.from_pretrained('bert-base-uncased')(test)[0] 
    # create the embeddings; grab [0] as BertModel returns a tuple
# print(encodings.size()) [10, 100, 768] passed into the LSTM
# print(encodings[0].size()) [100, 768] is the embedding of one comment
# print(encodings[0]) print out the actual embedding, a tensor