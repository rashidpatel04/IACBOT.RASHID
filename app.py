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
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', download_dir=nltk_data_dir)

load_dotenv()  # Load environment variables from the .env file

# Replace hard-coded keys with environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

# Initialize Flask app
app = Flask(__name__)

# Load chatbot resources
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.load(open('intents.json'))
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Helper functions for text processing and prediction
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

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
            engine="GPT-4",  # Use GPT-3.5 or GPT-4 as needed
            prompt=user_input,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return "Sorry, I couldn't process that. Please try again later."

def get_response(intent_list, intents_json, user_input=None):
    if intent_list:
        tag = intent_list[0]['intent']
        for i in intents_json['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    # Custom fallback response
    return "I'm sorry, I didn't understand that. Can you please rephrase?" if user_input else get_openai_response(user_input)

# Function to speak the response using gTTS
def speak_response(response):
    try:
        tts = gTTS(text=response, lang='en')
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        return None

# Route for handling voice input (Speech-to-Text)
@app.route('/voice', methods=['GET'])
def voice_input():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Listening for your voice input...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        user_input = recognizer.recognize_google(audio)
        print(f"Recognized text: {user_input}")

        intents_list = predict_class(user_input)
        response = get_response(intents_list, intents, user_input=user_input)

        # Generate audio file
        audio_file = speak_response(response)
        if audio_file:
            return jsonify({'response': response, 'audio_file': audio_file}), 200
        else:
            return jsonify({'response': response}), 200

    except sr.UnknownValueError:
        return jsonify({'response': "Sorry, I could not understand the audio."}), 500
    except sr.RequestError as e:
        return jsonify({'response': f"Sorry, there was an error with the speech recognition service: {e}"}), 500

# Route for web chat (Text Input)
@app.route('/chat', methods=['POST'])
def chat():
    try:
        incoming_message = request.json.get('message').strip()
        print(f"Incoming web message: {incoming_message}")

        intents_list = predict_class(incoming_message)
        response = get_response(intents_list, intents, user_input=incoming_message)

        return jsonify({'response': response}), 200
    except Exception as e:
        print(f"Detailed error in web chat: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'response': f'Error: {str(e)}'}), 500

# Route for WhatsApp integration via Twilio
@app.route("/webhook", methods=['POST'])
def whatsapp_webhook():
    try:
        incoming_message = request.form.get('Body')
        from_number = request.form.get('From')

        print(f"Incoming WhatsApp message from {from_number}: {incoming_message}")

        intents_list = predict_class(incoming_message)
        response = get_response(intents_list, intents, user_input=incoming_message)

        client.messages.create(
            body=response,
            from_=TWILIO_WHATSAPP_NUMBER,
            to=from_number
        )

        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Error in WhatsApp webhook: {e}")
        return jsonify({"status": "failure"}), 500

# Route for rendering the web app
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
