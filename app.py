from flask import Flask, request, jsonify, render_template
import random
import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import nltk
import openai
import speech_recognition as sr  # For speech-to-text conversion
from twilio.rest import Client
from gtts import gTTS
import os
import tempfile
from dotenv import load_dotenv

# Set NLTK data path
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir=nltk_data_dir)

load_dotenv()  # Load environment variables from the .env file

# Environment variables for keys
openai.api_key = os.getenv("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

# Initialize Flask app
app = Flask(__name__)

# Load chatbot resources globally
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.load(open('intents.json'))
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Helper functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_openai_response(user_input):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=user_input,
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return "Sorry, I couldn't process that. Please try again later."

def get_response(intent_list, intents_json):
    if intent_list:
        tag = intent_list[0]['intent']
        for i in intents_json['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "I'm sorry, I didn't understand that."

def speak_response(response):
    try:
        tts = gTTS(text=response, lang='en')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        return None

@app.route('/chat', methods=['POST'])
def chat():
    try:
        incoming_message = request.json.get('message').strip()
        intents_list = predict_class(incoming_message)
        response = get_response(intents_list, intents)
        return jsonify({'response': response}), 200
    except Exception as e:
        print(f"Error in chat route: {e}")
        return jsonify({'response': f'Error: {str(e)}'}), 500

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
