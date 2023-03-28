from flask import Flask, jsonify, request
import torch

app = Flask(__name__)

# Load the trained model
model = torch.load("my_model.pt")

# Define a route for making predictions
@app.route("/predict", methods=["POST"])
def predict():
    # Get the input data from the request
    data = request.get_json()

    # Tokenize the input data
    tokens = tokenizer(data["review"])

    # Convert the tokenized input to a tensor
    tensor = torch.tensor(tokens)

    # Pad the tensor to a fixed length
    tensor = pad_sequence([tensor], padding_value=0, batch_first=True, max_len=100)

    # Make a prediction with the model
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        label = int(torch.round(output).item())

    # Return the predicted label as a JSON response
    return jsonify({"label": label})

if __name__ == "__main__":
    app.run()
