import torch
import transformers 
from transformers import BertModel, BertTokenizer
import torch.nn as nn

# With our BERT encoding and embeddings created, we need a way to decode 
# Research Sentence Classifications

class BERTandLSTM(nn.Module):
    def __init__(self, weights='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(weights)
        self.LSTM = nn.LSTM(input_size=768, hidden_size=256, bidirectional=True, batch_first=True)
        # each token is of size 768; we specify 512 features to be created
        # [10, 100, 768] --> [10, ,100, 512]
        self.dense = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        # utilize sigmoid to bring values between 0 and 1, while maintaining order to select best classification

    def forward(self, tokenized):
        # note that all our comments come pre-tokenized, of size 100
        with torch.no_grad():
            embeddings = self.bert(tokenized)[0]
            # creates a BERT embedding of the comment; comment is now of shape 100 x 768, instead of 100
            # take [0] to get the 100x768 embedding; we then have batch_size embeddings, since BertModel returns tuple
        _, (HiddenStates, CellStates) = self.LSTM(embeddings)
        # feed [10, 100, 768] into the LSTM
        out = self.dense(torch.cat((HiddenStates[0], HiddenStates[1]), dim=1))
        # concatenates these two vectors to form a vector of length 512; dim=1 so its simply 1x512
        # this provides us with a valid input into our linear layer 
        # out is now a vector of 10x1
        out = self.sigmoid(out)
        # apply sigmoid to bring all values between 0 and 1, maintain order of vector 
        return out
