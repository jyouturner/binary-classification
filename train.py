import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Define a custom dataset that loads movie reviews and their labels
class MovieReviewDataset(Dataset):
    def __init__(self, reviews, labels):
        self.reviews = reviews
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]
        return review, label

# Define a simple neural network with a single hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Define the training function
def train(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float().unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# Define the evaluation function
def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            running_loss += loss.item()
            predicted_labels = torch.round(outputs)
            total_correct += (predicted_labels == labels.unsqueeze(1)).sum().item()
    accuracy = total_correct / len(dataloader.dataset)
    return running_loss / len(dataloader), accuracy

# Load the movie review dataset
reviews = ["This movie was great", "This movie was terrible", "This movie was just okay", ...]
labels = [1, 0, 0, ...]
dataset = MovieReviewDataset(reviews, labels)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create dataloaders for the training and validation sets
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)

# Define the model, optimizer, and loss function
model = NeuralNet(input_size=100, hidden_size=50)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# Train the model
for epoch in range(10):
    train_loss = train(model, train_dataloader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_dataloader, criterion)
    print(f"Epoch {epoch + 1}: train loss = {train_loss:.3f}, val loss = {val_loss:.3f}, val acc = {val_acc:.3f}")
