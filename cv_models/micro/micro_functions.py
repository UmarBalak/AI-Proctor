import speech_recognition as sr
import wave
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def listen():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print('Listening...')
        r.pause_threshold = 1
        audio = r.listen(source) 

    try:
        print("Recognizing...")    
        query = r.recognize_google(audio, language='en')
        return query.lower()  # Convert text to lowercase
    except Exception as e:
        print("Error:", str(e))
        return None

def compare_files(text, paper):
    word_tokens_1 = word_tokenize(text)
    word_tokens_2 = word_tokenize(paper)

    # Remove stop words from both tokenized texts
    stop_words = set(stopwords.words('english'))
    filtered_tokens_1 = [w for w in word_tokens_1 if w.lower() not in stop_words]
    filtered_tokens_2 = [w for w in word_tokens_2 if w.lower() not in stop_words]

    # Calculate similarity by finding common words
    common_words = set(filtered_tokens_1) & set(filtered_tokens_2)
    similarity_percentage = (len(common_words) / max(len(filtered_tokens_1), len(filtered_tokens_2))) * 100

    return similarity_percentage

