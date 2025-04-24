# GigFinder

GigFinder is a Python-based chatbot that helps users find live music events across the UK by scraping data from Live Nation. Users can interact with the chatbot to specify their city, preferred date(s), and music genre, and receive a list of relevant upcoming gigs. This project combines natural language processing, web scraping, and text classification to deliver a conversational and dynamic event-finding experience.

## Features

- City-based search (UK only)
- Supports single date or date range input
- Intelligent genre classification using a fine-tuned SetFit model
- Web scraping via BeautifulSoup for real-time event data
- Conversational interface with friendly dialogue flow
- Graceful handling of missing data or no event matches

## How It Works

1. **User Input**: The chatbot prompts the user for a UK city, music genre, and a date or date range.
2. **Data Processing**:
   - City names are URL-encoded using `urllib.parse`.
   - Dates are parsed with regular expressions and formatted appropriately.
   - Genres are classified into Live Nation-compatible categories using a SetFit model trained on a small, curated dataset.
3. **Web Scraping**: Event data is retrieved and parsed from Live Nation’s website using BeautifulSoup.
4. **Output**: The chatbot displays a neatly formatted list of events including name, venue, and date.

## Installation

```
git clone https://github.com/yourusername/gigfinder.git
cd gigfinder
pip install -r requirements.txt
```

## Usage

```
python gigfinder.py
```

Follow the prompts to enter your city, genre, and dates. The chatbot will display upcoming music events based on your input.

## Limitations

- Limited to Live Nation UK events
- No support for artist name search
- Sub-genres and niche genres may be misclassified
- Misspelled inputs may result in no matches

## Future Improvements

- Add fuzzy matching to handle typos
- Expand genre dataset and improve SetFit accuracy
- Include clickable event links and pagination for long date ranges
- Integrate multiple event platforms for broader results

## Ethical Considerations

This tool was built in accordance with Live Nation’s terms of service. No user data is stored, and all web scraping is done respectfully.

## Acknowledgments

- BeautifulSoup – for HTML parsing
- SetFit by HuggingFace – for text classification
- ChatGPT – for assistance with regex, debugging, and grammar-checking
