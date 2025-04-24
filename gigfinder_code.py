import time
import requests
import random
import re
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from chatbot_base import ChatbotBase
import urllib.parse

import os
from sklearn.preprocessing import LabelEncoder
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss
from datasets import load_dataset


class MyChatbot(ChatbotBase):
    def __init__(self, name='MyChatbot'):
        #initilizes chat bot with the different variables/objects i will need to create throughout the code
        ChatbotBase.__init__(self, name)
        self.target_url = "https://www.livenation.co.uk/event/allevents?page=1&dateFrom={self.start_date}&dateTo={self.end_date}&location={self.city}&genres={self.genre}"  
        self.city = None
        self.genre = None
        self.start_date = None 
        self.end_date = None 
        self.conversation_is_active = True  

        #code below references Week 7A notebook
        self.session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        #if the user can't train my data set in its respective file, will train it in this file 
        #code below references Week 7A notebook
        try:
            self.model = SetFitModel.from_pretrained("ckpt/", local_files_only=True)
        except Exception as e:
            model_id = "sentence-transformers/all-mpnet-base-v2"
            self.model = SetFitModel.from_pretrained(model_id)
            self.training_model()

        #genres on livenation that i wll have to choose from for my URL
        self.genres = [
        "afrobeats",
        "alternative and indie",
        "arts and culture",
        "comedy",
        "country",
        "electronic",
        "hard Rock/metal",
        "hip-hop/rap",
        "latin",
        "other",
        "pop",
        "reggae",
        "rock",
        "r&b/soul",
        "sport" ]
    
    
    #trains my data set for genres a user could input and their match on livenation
    def training_model(self):
        #code for this function references Week 7A notebook
        dataset = load_dataset('csv', data_files={
        "train": '/Users/clairereynolds/Documents/reynolds-NLP-chatbot/genre_datasets/genre_dataset.csv',  # Path to the training data
        "test": '/Users/clairereynolds/Documents/reynolds-NLP-chatbot/genre_datasets/test_genre_dataset.csv'  # Path to the test data
        })

        
        le = LabelEncoder()
        intent_dataset_train = le.fit_transform(dataset["train"]['label'])
        dataset["train"] = dataset["train"].remove_columns("label").add_column("label", intent_dataset_train).cast(dataset["train"].features)

        intent_dataset_test = le.fit_transform(dataset["test"]['label'])
        dataset["test"] = dataset["test"].remove_columns("label").add_column("label", intent_dataset_test).cast(dataset["test"].features)

        # setup and train classififer

        trainer = SetFitTrainer(
            model=self.model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            loss_class=CosineSimilarityLoss,
            metric="accuracy",
            batch_size=64,
            num_iterations=20,
            num_epochs=2,
            column_mapping={"text": "text", "label": "label"}
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        evaluation_results = trainer.evaluate()

        os.makedirs('ckpt/', exist_ok=True)

        trainer.model._save_pretrained(save_directory="ckpt/")

        #mapping and labels for the live nation genres 
        class_label_map = {
            0: "afrobeats",
            1: "alternative and indie",
            2: "arts and culture",
            3: "country",
            4: "electronic",
            5: "hard rock/metal",
            6: "latin",
            7: "pop",
            8: "reggae",
            9: "rock",
            10: "r&b/ soul",
            11: "hip-hop/rap",
        }



    #function for the beginning of my chatbot and where i gather my information for location, dates, and genre. then i sent that collected info to function for url formatting
    def greeting_user_inputs(self):
        print(f"Hey! GigFinder here, ready to help you find the perfect music gig for you to attend in the United Kingdom! \n")
        city_response = input("To start out, what UK city should I look in?\n")
        self.city_for_url(city_response)
        genre_response = input("Now, what genre are you looking for? \n")
        self.genre_for_url(genre_response)
        date_response = input("Do you have a specific date or date range in mind? Enter your date like DD/MM/YYYY or DD/MM/YYYY - DD/MM/YYYY! \n")
        self.date_for_url(date_response)

    #function for when a user wants to find more gigs, similar to greeting but language changes so it flows better and sounds more human like
    def ask_another(self):
        again=input("Do you want to find more gigs? \n").lower()

        if again == "yes":
            city_response = input("What UK city should I look in now?\n")
            #print(city_response)
            self.city_for_url(city_response)
            #print(self.city)
            genre_response = input("What genre do you want? \n")
            self.genre_for_url(genre_response)
            date_response = input("Are there any specific dates in mind? Remember, to put your date like this -> DD/MM/YYYY! \n")
            self.date_for_url(date_response)
            return self.generate_response()
    
        elif again == "no":
            return self.farewell()

        else:
            print("Sorry, I didn't catch whether you said yes or no.")
            return self.ask_another()
        
    
    #function webscraping notebook for initlizising beautiful soup   
    def get_html_content(self, target_url):
        #code below references Week 7A notebook
        try:
            r = self.session.get(target_url)
            soup = BeautifulSoup(r.content, 'html.parser')
            return soup
        except Exception as e:
            return None

    #takes the date input and formats it for the URL
    def date_for_url(self, date_response):
        #regex created with help from ChatGPT 
        date_range_regex = r"\d{1,2}\/\d{1,2}\/\d{2,4}\s*\-\s*\d{1,2}\/\d{1,2}\/\d{2,4}" #01/34/6789-12/45/789
        date_range_match = re.search(date_range_regex, date_response)

        #regex created with help from ChatGPT 
        single_date_regex = r"\d{1,2}\/\d{1,2}\/\d{2,4}"
        single_date_match = re.search(single_date_regex, date_response)

        start_date = None
        end_date = None

        if date_range_match:
            date = date_range_match.group()
            start_date = f"{date[6:10]}-{date[3:5]}-{date[0:2]}"
            end_date = f"{date[-4:]}-{date[16:18]}-{date[13:15]}"
            #end_date = f"{date[-4:]}-{date[14:16]}-{date[11:13]}"
            
   
        elif single_date_match:
            date = single_date_match.group()
            start_date = f"{date[-4:]}-{date[3:5]}-{date[0:2]}"
            end_date = start_date  # For a single date, the start and end are the same
           

        else:
            print("Sorry, I didn't catch that. Please make sure your dates are in this format -> DD/MM/YYYY or DD/MM/YYYY - DD/MM/YYYY.")
        
        self.start_date = start_date
        self.end_date = end_date
    
    #takes the city input and formats it for the URL
    def city_for_url(self, city_response):
        city_capitalize = city_response.title()
        #for citys that have - or , in the name this url library puts into right format
        city_url = urllib.parse.quote(city_capitalize)
        self.city = city_url

    
    #takes the genre input, matches it with the data set, and formats it for the URL
    def genre_for_url(self, genre_response):

        genre_lower = genre_response.lower()
        #output code references Week 7A notebook
        output = self.model.predict(genre_lower)
        output_label = int(output)
        genre_url = None

        if output_label == 0:
            genre_url = "afrobeats"   
        elif output_label == 1:
            genre_url = "alternative-and-indie"    
        elif output_label == 2:
            genre_url = "arts-and-culture" 
        elif output_label == 3:
            genre_url = "country" 
        elif output_label == 4:
            genre_url = "electronic" 
        elif output_label == 5:
            genre_url = "hard-rock-and-metal" 
        elif output_label == 6:
            genre_url = "latin"
        elif output_label == 7:
            genre_url = "pop"
        elif output_label == 8:
            genre_url = "reggae"
        elif output_label == 9:
            genre_url = "rock"
        elif output_label == 10:
            genre_url = "hip-hop-and-rap"
        elif output_label == 11:
            genre_url = "rnb-and-soul"
        elif output_label == 12:
            genre_url = "other"
        else:
            return "Sorry, didn't catch that. Try a different genre, like " + random.choice(self.genres) + "!"

        self.genre = genre_url
    
    #formatted url for webscraping with the inputs from the user, and the different HTML bits i need to dispay to the user
    def scrape_data(self):      
        
        url_date_range = f"https://www.livenation.co.uk/event/allevents?page=1&dateFrom={self.start_date}&dateTo={self.end_date}&location={self.city}&genres={self.genre}"
        self.target_url = url_date_range
        #print(self.target_url)
        #output code references Week 7A notebook
        response = self.session.get(url_date_range)
      
       # response code references ChatGPT troubleshooting suggestion
        if response.status_code == 200: 
            soup = BeautifulSoup(response.text, 'html.parser')
            
            no_results = soup.find('p', class_='allevents__noresults')
            if no_results:
                print("Sorry, I couldn't find any gigs with those requests. Try a wider date range, different genre, or different city!")

            else:
                event_listings = soup.find_all('li', class_='allevents__eventlistitem')
                print("Awesome! Here are some gigs I found just for you: \n")

                for event in event_listings:
                    event_name = event.find("span", class_ = "result-info__localizedname")
                    event_name_display = event_name.get_text(strip=True)

                    event_location = event.find("h4", class_ = "result-info__venue")
                    event_location_display = event_location.get_text(strip=True)

                    event_weekday = event.find("span", class_ = "event-date__date__weekday")
                    event_weekday_display = event_weekday.get_text(strip=True)

                    event_date = event.find("span", class_ = "event-date__date__day")
                    event_date_display = event_date.get_text(strip=True)

                    event_month=event.find("span", class_ = "event-date__date__month")
                    event_month_display = event_month.get_text(strip=True)

                    event_year= event.find("span", class_ = "event-date__date__year")
                    event_year_display = event_year.get_text(strip=True)

                    print (f" Event Name: {event_name_display}")
                    print (f" Location Name: {event_location_display}")
                    print (f" Date: {event_weekday_display} {event_date_display} {event_month_display} {event_year_display}" )
                    print ("---------------------------------------------------------------------------------")

    #function for saying goodbye
    def farewell(self):
        """Sends a farewell message to the user."""
        farewell_responses = ['Have fun!', 'Hope you found a good gig to check out!', "Don't forget go buy your ticket!"]
        random_farewell = random.choice(farewell_responses)
        print(random_farewell)
        self.conversation_is_active = False 

    #flow for chatbot to scrape data and ask another 
    def generate_response(self):
        self.scrape_data()
        self.ask_another()





# Example of chatbot interaction
if __name__ == "__main__":
    chatbot = MyChatbot()
    chatbot.greeting_user_inputs() 

    while chatbot.conversation_is_active:  # Keep the conversation going as long as it's active
        chatbot.generate_response()
        break
        # Check if the conversation is still active
        if not chatbot.conversation_is_active:
            break  # Exit the loop when conversation ends
