import pandas as pd
import re
import stanza
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the Finnish ScandiSent data
data = pd.read_csv('/Users/noraheikkila/ylesent/data/raw/fi.csv')
#print(data.head())
#print(data.info())

# Basic text cleaning
data['text'] = data['text'].str.lower()                                          # lowercase the data
data.drop_duplicates(inplace=True)                                               # drop the duplicatess and modify the existing df
data = data[data['text'].notna()]                                                # clear the missing values
data['text'] = data['text'].apply(lambda x: re.sub(r'[^a-zA-ZäöåÄÖÅ\s]', '', x)) # remoce from special characters, except letters and spaces
data['text'] = data['text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())      # normalize the spacing, so there is not double spacing
print(data.head())

# Initialize Stanza pipeline for Finnish tokenization and lemmatization
stanza.download('fi')
nlp = stanza.Pipeline(lang='fi', processors='tokenize,mwt,lemma')

# Function to tokenize and lemmatize text
def tokenize_lemmatize(data):
    doc = nlp(data)                                                      # runs the text through the finnish NLP pipeline
    return [word.lemma for sent in doc.sentences for word in sent.words] # loops through list of sentences in the text, list of objects in each sentence and collects the lemma of each word

data['tokens'] = data['text'].apply(tokenize_lemmatize) #list of lemmatized tokens for each text sample
print(data.head())

# Remove the stopwords
nltk.download('stopwords')
finnish_stopwords = set(stopwords.words('finnish'))

data['tokens'] = data['tokens'].apply(lambda tokens: [t for t in tokens if t not in finnish_stopwords])
print(data.head())

# Join the tokens into a string for vectorization (TF-IDF expects a string)
data['text_for_model'] = data['tokens'].apply(lambda tokens: " ".join(tokens))

print(data.head())

data[['text_for_model', 'label']].to_csv('/Users/noraheikkila/ylesent/data/preprocessed/training_preprocessed.csv', index=False)

