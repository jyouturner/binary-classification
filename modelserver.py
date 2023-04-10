import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
import nltk
import flask
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

# Download the NLTK stopwords and Punkt tokenizer
nltk.download("stopwords")
nltk.download("punkt")

# Prepare the data
data = pd.read_csv("job_application_rejections.csv")
X = data["text"]
y = data["label"].apply(lambda x: 1 if x == "reject" else 0)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the text data
def preprocess_text(text):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if w.lower() not in stop_words]

    return " ".join(filtered_text)

X_train = X_train.apply(preprocess_text)
X_test = X_test.apply(preprocess_text)

# Tokenize the text data
tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

# Build vocabulary
vocab = build_vocab_from_iterator(yield_tokens(X_train), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Numericalize the text data
def text_pipeline(text):
    return vocab(tokenizer(text))

def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(torch.tensor(_label, dtype=torch.float))
        text_list.append(torch.tensor(_text, dtype=torch.long))
    text_tensor = pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"])
    label_tensor = torch.tensor(label_list, dtype=torch.float)
    return text_tensor, label_tensor

train_data = [(text_pipeline(text), label) for text, label in zip(X_train, y_train)]
test_data = [(text_pipeline(text), label) for text, label in zip(X_test, y_test)]

# Create DataLoader objects
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_iterator = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_iterator = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# Build the model
class Classifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text)
        pooled = embedded.mean(dim=1)
        hidden = torch.relu(self.fc1(pooled))
        output = self.fc2(hidden)
        return self.sigmoid(output)

INPUT_DIM = len(vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1

model = Classifier(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

# Train the model
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()

model = model.to(device)
criterion = criterion.to(device)

def binary_accuracy(predictions, labels):
    rounded_predictions = torch.round(predictions)
    correct = (rounded_predictions == labels).float()
    accuracy = correct.sum() / len(correct)
    return accuracy


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch[0]).squeeze(1)
        loss = criterion(predictions, batch[1])
        acc = binary_accuracy(predictions, batch[1])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch[0]).squeeze(1)
            loss = criterion(predictions, batch[1])
            acc = binary_accuracy(predictions, batch[1])
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# Train the model
NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, test_iterator, criterion)

    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\tVal. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')

# Save the model
torch.save(model.state_dict(), 'model.pt')

# Flask API
app = flask.Flask(__name__)

def text_to_tensor(text):
    processed_text = preprocess_text(text)
    tensor = torch.tensor(text_pipeline(processed_text), dtype=torch.long)
    return tensor.unsqueeze(0)  # Add batch dimension


@app.route("/predict", methods=["POST"])
def predict():
    # Get the text string from the request body.
    text = flask.request.get_json()["text"]

    # Preprocess and convert the text string to a tensor
    preprocessed_text = preprocess_text(text)
    text_tensor = text_to_tensor(preprocessed_text)

    # Get the prediction from the model
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    prediction = model(text_tensor).round().item()

    # Return the prediction
    return flask.jsonify({"prediction": "reject" if prediction == 1 else "not reject"})

if __name__ == "__main__":
    app.run(debug=True)

