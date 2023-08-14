#Importing modules or libraries#
import discord
import os
import random
import openai
from dotenv import main
from dotenv import load_dotenv
from discord.ext import commands
from discord import app_commands
from discord import Interaction

#This is for initializing variables#

load_dotenv()

intents = discord.Intents.all()
client = commands.Bot(command_prefix="!",intents=discord.Intents.all())

#tokens#
serverID=1015179459244064818
token="MTA4NjY5Nzk1NDUwNjcxMTI0MA.GsxxOK.wJn33G6YbUTCzdgxobNZquXJHrjjTu_z0x7SnE"
openai.api_key = "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"

#Gives welcome enter when online#
@client.event
async def on_ready():
    print("Logged in as a bot {0.user}".format(client))
    print("Welcome to my world".format(client))
    
#responds to a command# 
@client.command()
async def hola(ctx):
    await ctx.send('hola how are you?')
    return
#responds to message on a specific channel#
@client.event
async def on_message(message):
    username = str(message.author).split("#")[0]
    channel = str(message.channel.name)
    user_message = str(message.content)
  
    print(f'Message {user_message} by {username} on {channel}')

    if message.author == client.user:
        return
    if channel == "testing":
        if user_message.lower() == "hello" or user_message.lower() == "hi":
              await message.channel.send(f'Hola {username}')
              return
        elif user_message.lower()== "to claire":
              await message.channel.send(f'to claire!!!')
        elif user_message.lower() == "bye":
              await message.channel.send(f'Chau {username}')
              

    
    #elif user_message.lower() == "tell me a joke":
            #jokes = [" Can someone please shed more\
            #light on how my lamp got stolen?",
                     #"Why is she called llene? She\
                     #stands on equal legs.",
                     #"What do you call a gazelle in a \
                     #lions territory? Denzel."]
            #await message.channel.send(random.choice(jokes))
        

# Set up OpenAI API credentials


# Define the OpenAI API parameters
#model = "text-davinci-002"
#temperature = 0.5
#max_tokens = 50
#stop = "\n"

#Initializing the bot#
#@client.event
#async def on_ready():
#    print(f'{client.user} has connected to Discord!')

#@client.command()
#async def ping(ctx):
#  await ctx.send('Pong!')
#@client.event
#async def on_message(message):
   
# Define the prompt for generating text with the OpenAI API
#    prompt = f"User message: {user_message}\nAI response:"
#    channel = str(message.channel.name)
# Use OpenAI to generate a response
#    if channel == "testing":
#        if user_message.lower() == client.user:
 #           response == openai.Completion.create(    
  #          model==model,
#         prompt==prompt,
#            temperature==temperature,
#            max_tokens==max_tokens,
#            stop==stop
#            )
# Send the response back to the user
 #           output= response.choices[0].text.strip()
  #          print(output)
   #         return
client.run(token)

