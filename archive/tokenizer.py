# Load the tokenizer from JSON
with open("tokenizer.json") as json_file:
    tokenizer_data = json_file.read()
tokenizer = tokenizer_from_json(tokenizer_data)

print("Tokenizer loaded successfully.")

import json
from keras.preprocessing.text import tokenizer_from_json

# Assuming tokenizer_data is already a dictionary:
# If loaded from JSON, it would be:
# with open('tokenizer.json', 'r') as f:
#     tokenizer_data = json.load(f)

# Convert the dictionary to a JSON string
tokenizer_json = json.dumps(tokenizer_data)

# Pass the JSON string to tokenizer_from_json
tokenizer = tokenizer_from_json(tokenizer_json)