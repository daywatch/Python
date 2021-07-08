# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:42:46 2020

@author: Tao Lin
I revised the code from pytorch tutorial
"""


import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import random
import math
from torchtext import data
from torchtext.data import Field, BucketIterator
import spacy
from nltk.translate.bleu_score import SmoothingFunction,sentence_bleu
from rouge_score import rouge_scorer
import time
from nltk.corpus import stopwords


if torch.cuda.is_available():
    device = torch.device('cuda')
else: device = torch.device('cpu')

#free your gpu memory
torch.cuda.empty_cache()

#spacy_en = spacy.load('en')
# WIN: change
spacy_en = spacy.load(r"C:\Users\25335\anaconda3\Lib\site-packages\en_core_web_sm\en_core_web_sm-2.3.0")
#spacy_en = spacy.load('en_core_web_sm')
def tokenize(text):
    return [example.text for example in spacy_en.tokenizer(text)]

#build fields and add sos and eos
SOURCE = data.Field(tokenize = tokenize, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True,
            batch_first = True,#new: avoid vertical transpositions
            fix_length = 1500) #char windows is 2000
TARGET = data.Field(tokenize = tokenize, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True,
            batch_first = True,#new
            fix_length = 300)#force the output to have the max length, useful? 

fields = {'source': ('source', SOURCE), 'target': ('target', TARGET)}

train_data, valid_data, test_data = data.TabularDataset.splits(
                                        #path = r'C:\Users\25335\Downloads\summarization_new\data\abstractive\5000', #5k
                                        #path = r'C:\Users\25335\Downloads\summarization_new\data\abstractive', # 10k
                                        path = r'C:\Users\25335\Downloads\COVID_summarizer\data\full', #3k source-15 + 5000
                                        train = 'train_full.json', 
                                        validation = 'valid_full.json',
                                        test = 'test_full.json',
                                        format = 'json',
                                        fields = fields)
#building dictionaries
SOURCE.build_vocab(train_data,min_freq=2)
TARGET.build_vocab(train_data,min_freq=2)

print(f"{len(train_data.examples)} training examples")
print(f"{len(valid_data.examples)} validation examples")
print(f"{len(test_data.examples)} testing examples")
print('\n')

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = 10,
    sort=False,
    device = device)


# encoder-decoder transformer
class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 3000):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len]
        #src_mask = [batch size, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
        
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        #src = [batch size, src len, hid dim]
            
        return src

class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, src len]
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src
    
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x

class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = 500):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, trg len]
        #src_mask = [batch size, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, trg len]
            
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
                
        #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)
        
        #output = [batch size, trg len, output dim]
            
        return output, attention

class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, trg len]
        #src_mask = [batch size, src len]
        
        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
            
        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention

class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention


INPUT_DIM = len(SOURCE.vocab)
OUTPUT_DIM = len(TARGET.vocab)

HID_DIM = 128 #256
ENC_LAYERS = 2
DEC_LAYERS = 2
ENC_HEADS = 2 #8
DEC_HEADS = 2 #8
ENC_PF_DIM = 2000
DEC_PF_DIM = 2000
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)

SRC_PAD_IDX = SOURCE.vocab.stoi[SOURCE.pad_token]
#TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
TRG_PAD_IDX = TARGET.vocab.stoi[TARGET.pad_token]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
        
model.apply(initialize_weights);


LEARNING_RATE = 0.0005

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        

        src = batch.source
        trg = batch.target
        
        optimizer.zero_grad()
        
        output, _ = model(src, trg[:,:-1])
                
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
            
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
            
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.source
            trg = batch.target

            output, _ = model(src, trg[:,:-1])

            
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            
            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 25
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


# prediction function
t = vars(test_data.examples[1])['source'] #list of input words from the testing pool
#print(" ".join(t))
#print('\n')

def make_summary(text_list, src_field, trg_field, model, device, max_len = 50):
    
    model.eval()
        
    #need a tokenization step for the txt input!

    text_list = [src_field.init_token] + text_list + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in text_list]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:], attention

#example_idx = 8


#trg = vars(test_data.examples[1])['target']
#print(f'trg = {trg}')

sum1, attention = make_summary(t,SOURCE, TARGET, model, device)

print(f'predicted trg = {sum1}')


# make predictions in batches for ROGUE metrics
gold = []
for item in test_data.examples:
    gold.append(vars(item)['target'])

pred = []
for i, batch in enumerate(test_data.examples):
    test_src = vars(test_data.examples[i])['source']
    sum_output,att = make_summary(test_src,SOURCE, TARGET, model, device)
    pred.append(sum_output)

# collect test results   
#f=open('result.txt','w', encoding='utf-8')
rouge1 = []
rougeL = []
for i in range(len(pred)):
    print(f"Gold: {vars(test_data.examples[i])['target'][1:]} \n")
    #f.write(f"Gold: {vars(test_data.examples[i])['target']} \n")
    
    print(f"Prediction: {pred[i]}")
    #f.write(f"Prediction: {pred_new[i]} \n")
    
    #print(f'BLEU IS {BLEU(pred_new[i],gold[i])} \n')
    #f.write(f'BLEU IS {BLEU(pred_new[i],gold[i])} \n')
    
    
    #pred_new = remove_stopwords(result.split())
    #gold_new = remove_stopwords(target[i].split())
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    score_total = scorer.score(" ".join(gold[i])," ".join(pred[i]))
    rouge1.append(score_total['rouge1'][1])
    rougeL.append(score_total['rougeL'][1])
    print(f"ROUGE1 is {score_total['rouge1']} \n")
    #f.write(f"ROUGE1 IS {score_total['rouge1']} \n")
    
    print(f"ROUGE1 is {score_total['rougeL']} \n")
    #f.write(f"ROUGE1 IS {score_total['rougeL']} \n\n\n")
    
#f.close()
print(f'Overall ROUGE1 is {sum(rouge1)/float(len(rouge1))}')
print(f'Overall ROUGEL is {sum(rougeL)/float(len(rougeL))}')
    
# SINGLE text input prediction
model.load_state_dict(torch.load(r"C:\Users\25335\Downloads\COVID_summarizer\transformer\model.pt"))

# sample txt
text = ("""London (CNN)The UK's troubled response to the coronavirus pandemic became even more confused on Monday, as government guidance seemingly at odds with itself rolled out across England, pushing the four nations of the UK further apart.
From Monday, employers in England can ask staff to return to workplaces if they believe they are Covid-19 safe.
When the policy was announced by Prime Minister Boris Johnson last month, he was accused of "passing the buck on this big decision to employers" by the UK's Trades Union Congress (TUC). The TUC criticized the government for announcing such a move while the country's widely-criticized test and trace infrastructure was "still patchy."
Also launching on Monday is an eye-catching government scheme, "Eat Out to Help Out," aimed at getting restaurants back on their feet as the UK's furlough scheme comes to an end. Throughout August, people dining out from Monday to Wednesday are to be offered a 50% discount -- limited to £10 ($13) per person, and not including alcoholic drinks.
Both policies are part of a wider plan to get the UK's economy moving after months of lockdown kept many Brits stuck indoors and working from home while businesses in the hospitality industry that relied on their custom were forced to stop operating.
But these moves to soften coronavirus prevention measures, in order to restart the economy, come at the same time as cases are rising in Europe.
Last week, Johnson himself warned that "the risk is starting to bubble up again," on the continent, adding: "I'm afraid you are starting to see in some places the signs of a second wave of the pandemic."
In the UK several localized mini-lockdowns have been implemented, as fears of that second wave move from possible to probable.
These localized containment measures have created a particularly stark situation in areas like Manchester, where it is now against the rules to meet family members in a back garden, and yet absolutely fine, according to the rules, to go to a restaurant.
The move marks a significant shift towards prioritizing economic recovery ahead of other issues, including public health.
'I was too fat,' Boris Johnson says in UK launch to tackle obesity
This might worry some who heard Chris Whitty, England's chief medical officer, last week say that trade-offs over what can be opened have "probably reached near the limit" of what is possible.
Graham Medley, a member of the UK government's Scientific Advisory Group for Emergencies (SAGE), suggested over the weekend that one such trade-off could be pubs for schools.
After months of remote learning, many parents are desperate to send their youngsters back into the classroom next month. But while they might expect children's education to be prioritized ahead of people enjoying a few pints at the local bar, the current trend for favoring business suggests they may be disappointed.
A government spokesperson said on Monday that they expect all schools to be open from September. They stressed, however, that the UK's response would continue to be localized, where "you would assess the situation on the ground and take whatever steps were required to slow the spread of the virus."
On the specific question of choosing between pubs and schools, the spokesperson said, "we are committed to supporting the hospitality industry which has had a very tough time."
These measures apply only to England, as public health policy is a matter for the devolved governments of Scotland, Wales and Northern Ireland.
Throughout the pandemic, Johnson's government has been criticized by politicians and leaders in the three other nations for his perceived recklessness, most notably Scotland's First Minister, Nicola Sturgeon.
This perception of England mishandling the crisis has led to a surge for support in Scottish independence north of the border, though it is worth noting that Scotland's coronavirus record is not that much better than England's -- for every 100,000 people, 77 in Scotland have died with Covid-19 listed on their death certificate, versus 86 in England.
"""
)

def make_single_summary(text, src_field, trg_field, model, device, max_len_in = 1500, max_len_out = 50):
    
    model.eval()
    
    #need a tokenization step for the txt input!
    text_list = text.split()[0:max_len_in] 

    text_list = [src_field.init_token] + text_list + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in text_list]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len_out):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:], attention

sample_result = make_single_summary(text, SOURCE, TARGET, model, device)
print(sample_result[0])