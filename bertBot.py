import discord
import os 
from transformers import BertTokenizer
import torch
from dotenv import load_dotenv

model = torch.load('bot__model.pt')
bertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def prepMessage(message):
    tokenized = torch.tensor(bertTokenizer.encode(message, padding='max_length', max_length = 100, truncation=True))
    return tokenized.view(1, 100)

def filterMessage(message):
    model.eval()
    tokenized = prepMessage(message)
    output = model(tokenized)
    print(output.item())
    if output.item() > 0.5:
        print("FILTERED!")
        return True
    else:
        print("NOT FILTERED.")
        return False
        
load_dotenv()
path = 'data/token.env'
load_dotenv(dotenv_path=path, verbose=True)
TOKEN = os.getenv("DISCORD_TOKEN")
print(TOKEN)

def run_discord_bot():
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    # initialize the discord "bot" itself with intents we specify

    # the client receives an event update from discord 
    @client.event
    async def on_ready():
        print(f'{client.user} is ready!')
        # when bot goes online, we print to console

    @client.event
    async def on_message(message):
        msg = str(message.content)
        if filterMessage(msg):
            await message.delete()
            # delete the message if the function tells us it is offensive/hate speech

    client.run(TOKEN)

run_discord_bot()