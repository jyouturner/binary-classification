# Define some new movie reviews
new_reviews = ["This movie was awesome", "This movie was terrible", "I hated this movie", ...]

# Tokenize the new reviews
new_tokens = [tokenizer(review) for review in new_reviews]

# Convert the tokenized reviews to tensors
new_tensors = [torch.tensor(token) for token in new_tokens]

# Pad the tensors to a fixed length
new_tensors = [pad_sequence([tensor], padding_value=0, batch_first=True, max_len=100) for tensor in new_tensors]

# Stack the tensors into a single batch
new_batch = torch.stack(new_tensors)

# Make predictions with the model
model.eval()
with torch.no_grad():
    outputs = model(new_batch)
    predicted_labels = torch.round(outputs)

# Print the predicted labels
for i, review in enumerate(new_reviews):
    print(f"{review}: {'positive' if predicted_labels[i] == 1 else 'negative'}")
