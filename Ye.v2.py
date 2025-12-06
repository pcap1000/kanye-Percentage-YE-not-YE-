#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import torch.optim as optim


# In[2]:


data = pd.read_csv('./data/YE_west.csv')


# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# In[4]:


data.head()


# In[5]:


for x in data['verse']:
    print(x)


# In[6]:


# data[i for i in label i == 1 ]


# In[7]:


def clean_string(txt):
    new_text = ''
    i = 0
    while i < len(txt):

        # stop when '(' is found
        if txt[i] == '(':
            break

        # skip http links
        if txt[i:i+4] == "http":
            # skip until a space or end
            while i < len(txt) and txt[i] != ' ':
                i += 1
            continue

        # skip punctuation
        if txt[i] in ['.', "'", '!','“','”', '@']:
            i += 1
            continue

        new_text += txt[i]
        i += 1

    return new_text


# In[8]:


clean_string("'Happy'! boy.(kumar)")


# In[9]:


clean_string("DEAR FUTURE, I STILL BELIEVE IN YOU PRINTED IN THE NEW YORK TIMES THIS MORNING https://t.co/3hGgcjHzRE https://t.co/7tlMR2wa0q")


# In[10]:


for x in range(len(data['verse'])):
    value = clean_string(data.loc[x, 'verse'])
    data.loc[x, 'verse'] = value


# In[11]:


data.loc[390, 'verse']


# In[12]:


y = data['label']


# In[13]:


print(f'Count of Ye:{(len([i for i in y if i==1]))}')
print(f'Count of NOT Ye:{(len([i for i in y if i==0]))}')


# In[14]:


X = data['verse']


# In[15]:


vocab = {'<UNK>':0}


# In[16]:


def build_vocab():
    string=''
    for x in data['verse']:
        string =string +' '+ x
    value = string.split()

    for x in value:
        if x not in vocab:
            vocab[x] = len(vocab)


# In[17]:


build_vocab()


# In[18]:


vocab


# In[19]:


def text_to_indices(txt):
    txt = txt.split()

    if len(txt) == 0:
        return [0]  # UNK token so seq_len = 1, NOT zero

    int_val = []
    for x in txt:
        if x in vocab:
            int_val.append(vocab[x])
        else:
            int_val.append(0)

    return int_val


# In[20]:


text_to_indices('I love music')


# In[21]:


class CustomDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(text_to_indices(self.data[idx]), dtype=torch.long),
            torch.tensor(self.label[idx], dtype=torch.float32)
        )


# In[22]:


X.head(1)


# In[23]:


dataCus = CustomDataset(X,y)
print(len(dataCus))
dataCus[10]


# In[24]:


datum = DataLoader(dataCus, batch_size=1,shuffle=True)


# In[25]:


torch.is_tensor(x)


# In[26]:


for x,y in datum:
    print(f'X : {x}{x.dtype}, Y : {y}{y.dtype}')


# In[27]:


class ModelYe(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,64)
        self.rnn = nn.RNN(64,128,batch_first=True)
        self.linear = nn.Linear(128,1)

    def forward(self, data):
        output = self.embedding(data)
        hidden, output = self.rnn(output)
        # hidden is the hidden_state, Which is not   requred for the model
        output = self.linear(output)
        return(output)



# In[28]:


model = ModelYe(len(vocab)).to(device)


# In[35]:


learning_rate = 0.001
epochs = 5


# In[30]:


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# In[31]:


# for epoch in range(epochs):
#     total_loss = 0
#     for x, y in datum:
#         output = model(x)
#         output = output.squeeze(2).squeeze(1)
#         # print(f'Model-out Shape :{output.dtype}')
#         # print(f'Y shape :{y.dtype}')
#         loss = criterion(output,y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f'Epoch:{epoch + 1}/{epochs}, Loss:{total_loss} ')


# In[36]:


for epoch in range(epochs):
    total_loss = 0
    for x, y in datum:
        model.train()
        x = x.to(device)
        output = model(x)
        output = output.squeeze(2).squeeze(1)
        # print(f'Model-out Shape :{output.dtype}')
        # print(f'Y shape :{y.dtype}')
        y = y.to(device)
        loss = criterion(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch:{epoch + 1}/{epochs}, Loss:{total_loss} ')


# In[37]:


def predict_Ye(text):
    text = clean_string(text)
    text_val = text_to_indices(text)

    input_tensor = torch.tensor(text_val, dtype=torch.long).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)
        if probs <= 0.1 : 
            value = "U NO Ye -" + ' Confidences:' + str(probs.item())
        else:
            value = 'YE % - '+ str(probs.item()*100)

    return value


# In[38]:


predict_Ye('loving god')


# In[39]:


predict_Ye('Dreams work only when you do')


# In[40]:


predict_Ye('Consistency is more important than short bursts of motivation.')


# In[41]:


predict_Ye('Shoot for the stars, so if you fall you land on a cloud')


# In[42]:


predict_Ye('''Everything I'm not makes me everything I am''')


# In[43]:


predict_Ye('''I have not failed. I've just found 10,000 ways that won't work.''')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




