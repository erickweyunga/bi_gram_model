import os
import random
import re
import string
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter, defaultdict
import pickle

# Download required NLTK resources
nltk.download('punkt', quiet=True)

# Configuration
INPUT_DATA_DIR = "../data"
MODELS_DIR = "./models"
OUTPUT_DIR = "./output"
SAMPLE_SIZE = 1000000  # Characters to process
SEQUENCE_LENGTH = 10  # Input sequence length for prediction
RANDOM_SEED = 42
BATCH_SIZE = 128
EPOCHS = 20
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set random seed for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Set device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def is_hidden(filepath):
    """Check if file is hidden."""
    return os.path.basename(filepath).startswith(".")


def clean_text(text):
    """Clean the text while preserving sentence structure and key formatting."""
    # Replace line breaks and excessive whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Keep only letters, numbers, spaces, and basic punctuation
    keep_chars = ".?!,;:()'\""
    cleaned_text = ""
    for char in text:
        if char.isalnum() or char.isspace() or char in keep_chars:
            cleaned_text += char
        else:
            cleaned_text += ' '

    # Fix spaces around punctuation
    for char in ".?!,;:)":
        cleaned_text = re.sub(rf'\s+{re.escape(char)}', char, cleaned_text)

    # Fix spaces before opening parentheses
    cleaned_text = re.sub(r'\s+\(', ' (', cleaned_text)

    # Remove excess spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text.lower()


def extract_sentences(text):
    """Extract well-formed sentences from text."""
    # Clean the text first
    cleaned_text = clean_text(text)

    # Try to extract complete sentences
    sentences = sent_tokenize(cleaned_text)
    result = []

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence.split()) >= 3:  # Only keep sentences with at least 3 words
            result.append(sentence)

    return result


def load_sample_data():
    """Load a representative sample of text data from files."""
    all_text = ""
    files_processed = 0
    total_chars = 0

    print("Loading data files...")
    for filename in os.listdir(INPUT_DATA_DIR):
        filepath = os.path.join(INPUT_DATA_DIR, filename)
        if not is_hidden(filepath) and total_chars < SAMPLE_SIZE:
            try:
                with open(filepath, "r", encoding="utf-8", errors='ignore') as f:
                    content = f.read()
                    if content.strip():
                        # Extract sentences
                        sentences = extract_sentences(content)
                        all_text += " ".join(sentences) + " "

                        # Update stats
                        total_chars += len(content)
                        files_processed += 1
                        if files_processed % 10 == 0:
                            print(f"Processed {files_processed} files, {total_chars} characters")

                        # Stop if we've reached the sample size
                        if total_chars >= SAMPLE_SIZE:
                            break
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

    print(f"Finished loading {files_processed} files. Total text length: {len(all_text)} characters")
    return all_text


class Vocabulary:
    """Vocabulary class to convert words to indices and vice versa."""

    def __init__(self):
        self.word_to_index = {"<PAD>": 0, "<UNK>": 1}
        self.index_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.word_counts = Counter()
        self.n_words = 2  # Start with padding and unknown tokens

    def add_word(self, word):
        self.word_counts[word] += 1

    def build_vocab(self, min_count=2):
        """Build vocabulary from word counts, including only words that appear at least min_count times."""
        for word, count in self.word_counts.items():
            if count >= min_count and word not in self.word_to_index:
                self.word_to_index[word] = self.n_words
                self.index_to_word[self.n_words] = word
                self.n_words += 1

    def word_to_idx(self, word):
        """Convert a word to its index in the vocabulary."""
        return self.word_to_index.get(word, self.word_to_index["<UNK>"])

    def idx_to_word(self, idx):
        """Convert an index to its corresponding word in the vocabulary."""
        return self.index_to_word.get(idx, "<UNK>")


class TextDataset(Dataset):
    """PyTorch Dataset for language modeling."""

    def __init__(self, text, vocab, sequence_length):
        self.text = text
        self.vocab = vocab
        self.sequence_length = sequence_length
        self.words = word_tokenize(text)
        self.word_indices = [vocab.word_to_idx(word) for word in self.words]

        # Create sequences and targets
        self.sequences = []
        self.targets = []

        for i in range(0, len(self.word_indices) - sequence_length):
            self.sequences.append(self.word_indices[i:i + sequence_length])
            self.targets.append(self.word_indices[i + sequence_length])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long)
        )


class LSTMLanguageModel(nn.Module):
    """LSTM-based language model."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(LSTMLanguageModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x shape: (batch_size, sequence_length)
        embeddings = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)

        # Pass through LSTM
        if hidden is None:
            lstm_out, hidden = self.lstm(embeddings)  # (batch_size, sequence_length, hidden_dim)
        else:
            lstm_out, hidden = self.lstm(embeddings, hidden)

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Only use the output from the last time step for next word prediction
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        # Project to vocabulary size
        output = self.fc(lstm_out)  # (batch_size, vocab_size)

        return output, hidden

    def init_hidden(self, batch_size, device):
        # Initialize hidden state and cell state
        return (
            torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device),
            torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        )


def train_model(model, train_loader, vocab_size, epochs, learning_rate, device):
    """Train the language model."""
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        hidden = None

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs, hidden = model(inputs)

            # Detach hidden states to prevent backpropagation through time steps
            hidden = tuple(h.detach() for h in hidden)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            # Apply gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

            # Print progress
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Print epoch statistics
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} completed, Average Loss: {avg_loss:.4f}")

    return model


def generate_text(model, vocab, seed_text, sequence_length, num_words=30, temperature=0.3, device=device):
    """Generate text using the trained model."""
    model.eval()  # Set to evaluation mode

    # Tokenize seed text
    words = word_tokenize(seed_text.lower())

    # Convert words to indices
    current_sequence = [vocab.word_to_idx(word) for word in words]

    # Pad if sequence is shorter than required
    if len(current_sequence) < sequence_length:
        padding = [vocab.word_to_idx("<PAD>")] * (sequence_length - len(current_sequence))
        current_sequence = padding + current_sequence

    # Truncate if sequence is longer than required
    if len(current_sequence) > sequence_length:
        current_sequence = current_sequence[-sequence_length:]

    # Start with seed text
    generated_text = seed_text

    # Generate new words
    for _ in range(num_words):
        # Convert to tensor and add batch dimension
        x = torch.tensor([current_sequence], dtype=torch.long).to(device)

        # Generate prediction
        with torch.no_grad():
            output, _ = model(x)

        # Apply temperature to output logits
        output = output.squeeze() / temperature
        probs = torch.softmax(output, dim=0).cpu().numpy()

        # Sample from the distribution
        next_word_idx = np.random.choice(len(probs), p=probs)
        next_word = vocab.idx_to_word(next_word_idx)

        # Skip padding and unknown tokens
        if next_word in ["<PAD>", "<UNK>"]:
            continue

        # Add new word to generated text
        generated_text += " " + next_word

        # Update current sequence
        current_sequence = current_sequence[1:] + [next_word_idx]

        # End if sentence marker and minimum length
        if next_word in ['.', '!', '?'] and len(generated_text.split()) > 15:
            if random.random() < 0.75:  # 75% chance to end
                break

    # Clean up the generated text
    generated_text = generated_text.strip()
    if generated_text:
        generated_text = generated_text[0].upper() + generated_text[1:]

    # Fix spacing around punctuation
    for char in ".?!,;:)":
        generated_text = re.sub(rf'\s+{re.escape(char)}', char, generated_text)

    return generated_text


def main():
    """Main function to train the neural language model and generate text."""
    model_filepath = os.path.join(MODELS_DIR, "pytorch_language_model.pt")
    vocab_filepath = os.path.join(MODELS_DIR, "pytorch_vocab.pkl")

    # Load and preprocess data
    text = load_sample_data()

    # Create vocabulary
    print("Building vocabulary...")
    vocab = Vocabulary()
    for word in word_tokenize(text):
        vocab.add_word(word)

    # Build vocabulary (include only words that appear at least twice)
    vocab.build_vocab(min_count=2)
    print(f"Vocabulary size: {vocab.n_words}")

    # Save vocabulary
    with open(vocab_filepath, 'wb') as f:
        pickle.dump(vocab, f)

    # Create dataset and dataloader
    dataset = TextDataset(text, vocab, SEQUENCE_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Create model
    model = LSTMLanguageModel(
        vocab_size=vocab.n_words,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    # Print model summary
    print(model)

    # Train model
    print("\nTraining model...")
    model = train_model(
        model=model,
        train_loader=dataloader,
        vocab_size=vocab.n_words,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device
    )

    # Save model
    print(f"Saving model to {model_filepath}...")
    torch.save(model.state_dict(), model_filepath)

    # Generate examples
    print("\nGenerating text examples...")
    seed_words = ["natural", "the", "market", "company", "bank", "stock", "trading", "dollar"]

    for seed in seed_words:
        for i in range(3):
            generated = generate_text(model, vocab, seed, SEQUENCE_LENGTH, num_words=50, device=device)
            print(f"\nStarting with '{seed}' (Example {i + 1}):")
            print(generated)

            # Save to file
            output_filepath = os.path.join(OUTPUT_DIR, f"pytorch_generated_{seed}_{i + 1}.txt")
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(generated)


if __name__ == "__main__":
    main()
