#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as ply
import string
import os
import re
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


# In[3]:


# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[5]:


print(len(sentences))
print(len(split_sentences))
print(len(split_labels))


# In[98]:


#EMOTION DECISION FUNCTIONS

def emo_4(emo):
    if (emo == 'Surprise' or emo == 'Excited'):
        emo = 'Happiness'
    elif (emo == 'Fear'):
        emo = 'Sadness'
    elif (emo == 'Disgust' or emo == 'Frustration'):
        emo = 'Anger'
    else:
        emo=emo

    return emo

def emo2_1(emo_duo):
    if isinstance(emo_duo, str):
        return emo_4(emo_duo)
    else:
        if emo_duo == ['Neutra', 'tate']:
            return 'Neutral'
        else:
            return emo_4(emo_duo[0])



def emo_dec(emo_list):
    h_count = 0
    a_count = 0
    s_count = 0
    n_count = 0
    o_count = 0


    for emo in emo_list:
        emo = emo2_1(emo)
        if emo == 'Neutral':
            n_count+=1
        elif emo == 'Sadness':
            s_count+=1
        elif emo == 'Happiness':
            h_count+=1
        elif emo == 'Anger':
            a_count+=1
        else:
            o_count+=1

    count_list = [n_count,s_count,h_count,a_count,o_count]
    dec_list = ['Neutral', 'Sadness', 'Hapiness', 'Anger', 'Other']

    for i,count in enumerate(count_list):
        if count>1:
            return dec_list[i]
    return 'Other'





# In[101]:


#GET EMOTION
emo_final = {}
emo_labels = []
emo_names = []
path = '../../EmoLabels/'  #WINDOWS PATH
emo_files = os.listdir(path)
emo_files.sort()
for f in emo_files:
    if (f[-3:] == 'txt'):
        em = open(path+f,"r")
        try:
            lines = em.readlines()
        finally:
            em.close

        for line in lines:
            em_split = line.split()
            if len(em_split)<= 3:
                emo_names.append(em_split[0])
                emo_labels.append(em_split[1][1:-1])
            elif len(em_split) == 4:
                emo_names.append(em_split[0])
                emo_labels.append([em_split[1][1:-1],em_split[2][1:-1]])


emo_unsort = zip(emo_names,emo_labels)
emo_sort = sorted(emo_unsort, key=lambda emo: emo[0])

group_emo = [ [emo_sort[0][0], [] ]]
for ev in emo_sort:
    if (ev[0] == group_emo[-1][0]):
        group_emo[-1][1].append(ev[1])
    else:
        group_emo.append([ev[0],[ev[1]]])
for emo_pair in group_emo:
    emo_pair[1] = emo_dec(emo_pair[1])
    emo_final[emo_pair[0]] = emo_pair[1]


# In[103]:





# In[115]:


#SEPERATE .txt FILES

path = '../../Transcriptions/'  #WINDOWS PATH
#path = '../Datsets/Transcriptions/'  #LINUX PATH
names = []
times = []
sentences =[]
labels = []
kelvin_labels =[]

transcripts = os.listdir(path)
transcripts.sort()
for txt in transcripts:
    dialog = open(path+txt,"r")
    try:
        lines = dialog.readlines()
    finally:
        dialog.close
    for line in lines:
        split_line = line.split()
        if len(split_line[0]) > 12:
            names.append(split_line[0])
            times.append(split_line[1][:-1])
            sentences.append(" ".join(split_line[2:]))
            try:
                kelvin_labels.append(emo_final[split_line[0]])
            except:
                continue

#SPLIT UP INDIVIDUAL SENTNECES
split_sentences = []
split_names = []
split_boys = zip(sentences,names)
for sent,name in split_boys:
    splitings = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', sent)
    for spliter in splitings:
        if len(spliter)>0:
            try:
                labels.append(emo_final[name])
                split_names.append(name)
                split_sentences.append(spliter)
            except:
                continue


        else:
            continue


# In[116]:


#CHECK IF LABELS AND SENTENCE LENGTH MATCH
print(len(split_sentences))
print(len(labels))


# In[118]:


print(len(kelvin_labels))


# In[ ]:


#BERT PREPROCESSING

token_tensors = []
segment_tensors = []
token_sentences = []
token_indices = []
token_match = []

for split_sent in split_sentences:
    #ADD CLS & SEP FOR EVERY INDIVIDUAL SENTENCE
    marked_sent = "[CLS] " + split_sent + " [SEP]"
    #TOKENIZATION
    token_sent = tokenizer.tokenize(marked_sent)
    token_sentences.append(token_sent)
    #INDEXING
    indexed_sent = tokenizer.convert_tokens_to_ids(token_sent)
    token_indices.append(indexed_sent)
    #SEGMENT ID
    seg_id = [1]*len(token_sent)
    token_match.append(zip(token_sent,indexed_sent))


    #TENSORS
    token_tensors.append(torch.tensor([indexed_sent]))
    segment_tensors.append(torch.tensor([seg_id]))




bert_input=zip(token_tensors,segment_tensors)



# In[ ]:





# In[ ]:


#LOAD PRETRAINED BERT
model = BertModel.from_pretrained('bert-base-uncased')
#TRANSFER LEARNING NO NEED FOR TRAINING -> ONLY FEED FORWARD
model.eval()


# In[ ]:


sentenes_feat = []
with torch.no_grad():
    for token_sent,segment_id in bert_input:
        encoded_layers, _ = model(token_sent, segment_id)
        sent_feat = []
        for i in range(len(token_sent)):
            token_feat = encoded_layers[-4:][0][i]
            for l in range(1,4):
                token_feat+=encoded_layers[-l:][0][i]
            sent_feat.append(token_feat)
        sentences_feat.append(sent_feat)


# In[ ]:


print((token_feat[0]))


# In[ ]:


print(sentences_feat[0][0])


# In[ ]:


a =torch.tensor([1,2,3])
b =torch.tensor([4,5,6])
print(a+b)


# In[9]:


emo_duo = ['Neutra', 'tate']

if emo_duo == ['Neutra', 'tate']:
    emo_1 = 'Neutral'
print(emo_1)


# In[ ]:




