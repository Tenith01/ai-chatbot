from flask import Flask, render_template, request, jsonify
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import random

app = Flask(__name__)

# Load intents and trained model
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE, map_location=torch.device('cpu'))

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Dr.Kasun"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train', methods=['POST'])
def train():
    # Add code to handle training data sent from the front-end
    # and train the model with the new data
    pass


@app.route('/chat', methods=['POST'])
def chat():
    sentence = request.form['text']
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                return jsonify({'response': response})
    else:
        print(jsonify({'response': f"{bot_name}: I do not understand..."}))
        return jsonify({'response': f"{bot_name}: I do not understand..."})

if __name__ == "__main__":
    app.run(debug=True, port=30034)

