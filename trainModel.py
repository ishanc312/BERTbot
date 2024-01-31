import torch 
import prepareData as p
import pandas as pd 
import bertLSTM as bl
import torch
import torch.nn as nn
import transformers
import time
import pathlib

# -------------------------- PREPPING OUR DATA 

df = pd.read_csv('data/HateSpeech.csv')
df = df[["tweet", "class"]]
# reducing the dataframe to the essentials
df = p.balanceData(df)
# balance the data utilizing oversampling
df = p.tokenizeColumn(df)
# tokenize all the tweets of the column
df = df.sample(frac=1)
# randomly sample 

# We need to balance this dataset, because there is a disproportionate amount of offensive/hate speech
# the "neither" category is very small, so we need to undersample + retrain the bot 
# what is occuring is that even "kind" messages are being read as offensive so we need to fix this issue 

df_train = p.splitSets(df, 0, 10000)
df_val = p.splitSets(df, 10000, 11000)
df_test = p.splitSets(df, 11000, 13000)
# Split into training, validation, test set
# shuffle all the data once more once we split 

trainBert = p.getTensor(df_train, "tweet")
valBert = p.getTensor(df_val, "tweet")
testBert = p.getTensor(df_test, "tweet")
# print(trainBert.size()) --> [30000, 100]

y_train = (p.getTensor(df_train, "class")).view(10000, 1)
y_val = (p.getTensor(df_val, "class")).view(1000,1)
y_test = (p.getTensor(df_test, "class")).view(2000,1)
# print(y_train.size()) --> [30000, 1]
# reshaping it so it serves as a valid input (before it was [10000, 3] so staying in line with that)

# ---------------------------- SETTING UP OUR MODEL

model = bl.BERTandLSTM()
n_epochs = 4
batch_size = 10
# batches of 10 commments
lr = 0.001
# learning rate

lossFN = nn.BCELoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)

train_losses = []
val_losses = []

# --------------------------------

def createBatch(x_in, y_in, batch_size):
    i = 0
    for i in range(0, len(x_in)-batch_size, batch_size):
        x_batch = x_in[i:i+batch_size]
        y_batch = y_in[i:i+batch_size]
        yield x_batch, y_batch.float()
    if i + batch_size < len(x_in):
        yield x_in[i+batch_size:], y_in[i+batch_size:].float()
    # Keeps returning batches of size 10, iterating over the full data set 
    # final if statement is to get the last batch 3
    # otherwise last batch gets skipped over because "stop" value of range is not inclusive

for n in range(n_epochs):
    i = 1
    j = 1
    model.train(True)
    train_loss = 0
    for x_batch, y_batch in createBatch(trainBert, y_train, batch_size):
        y_pred = model(x_batch)

        print(y_pred)
        print(f'Iteration {i}')
        i+=1
        
        optim.zero_grad()
        loss = lossFN(y_pred, y_batch)
        # create object of the criterion
        loss.backward()
        # backpropagate
        optim.step()
        train_loss = train_loss + loss.item()
        # add the loss, using .item() 
    train_losses.append(train_loss)
    # append the loss for the current epoch 

    val_loss = 0
    model.eval() # disables the dropout layer so we can evaluate the model properly 
    with torch.no_grad(): # deattaches the gradient calculations as we are evaluating, speeds up computations
        for x_batch, y_batch in createBatch(valBert, y_val, batch_size):
            y_pred = model(x_batch)
            loss = lossFN(y_pred, y_batch)
            val_loss = val_loss + loss.item()
            # validation, so we don't step the optimizer or back propagate 
            print(f'Iteration {j}')
            j+=1
        val_losses.append(val_loss)

    print(f'EPOCH {n} COMPLETE!')

print("DONE TRAINING")
torch.save(model, './bot__model.pt') 