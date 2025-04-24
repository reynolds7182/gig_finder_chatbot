# Import your chatbot class here
from chatbot_base import ChatbotBase
from my_chatbot import MyChatbot

import time
import requests
import urllib.request
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

if __name__ == "__main__":
    chatbot = MyChatbot()
    chatbot.greeting_user_inputs() 

    while chatbot.conversation_is_active:  
        chatbot.generate_response()
        break
        
        if not chatbot.conversation_is_active:
            
            break  



    
