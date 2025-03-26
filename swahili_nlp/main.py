import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import re
import numpy as np
import random
import time
import argparse
from collections import Counter
from tqdm import tqdm

# Configuration
MODELS_DIR = "./models"
SWAHILI_DATA_DIR = "../swahili_data"
EMBEDDING_DIM = 200  # Increased for better capturing of language features
HIDDEN_DIM = 256  # Increased for more capacity
NUM_LAYERS = 2
DROPOUT = 0.3  # Slightly increased to prevent overfitting
SEQUENCE_LENGTH = 20  # Increased for better context
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
MIN_WORD_FREQ = 2  # Minimum frequency for a word to be included in vocabulary
RANDOM_SEED = 42

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Set device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Ensure model directory exists
os.makedirs(MODELS_DIR, exist_ok=True)


class SwahiliPreprocessor:
    """Class for preprocessing Swahili text."""

    @staticmethod
    def clean_text(text):
        """Clean Swahili text while preserving language-specific characters."""
        # Replace line breaks and excessive whitespace
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        # Keep Swahili-specific characters and punctuation
        # Swahili uses standard Latin alphabet plus apostrophes for certain words
        keep_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .?!,;:()'\""
        cleaned_text = "".join(c for c in text if c in keep_chars)

        # Fix spaces around punctuation
        for char in ".?!,;:)":
            cleaned_text = re.sub(rf'\s+{re.escape(char)}', char, cleaned_text)

        # Fix spaces before opening parentheses
        cleaned_text = re.sub(r'\s+\(', ' (', cleaned_text)

        # Remove excess spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        return cleaned_text.lower()

    @staticmethod
    def tokenize(text):
        """Simple word tokenization for Swahili."""
        # Split text on spaces after cleaning
        return text.split()


class SwahiliVocabulary:
    """Vocabulary class for Swahili text."""

    def __init__(self):
        self.word_to_index = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.index_to_word = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"}
        self.word_counts = Counter()
        self.n_words = 4  # Start with special tokens

    def add_word(self, word):
        self.word_counts[word] += 1

    def build_vocab(self, min_freq=2):
        """Build vocabulary from word counts."""
        for word, count in self.word_counts.items():
            if count >= min_freq and word not in self.word_to_index:
                self.word_to_index[word] = self.n_words
                self.index_to_word[self.n_words] = word
                self.n_words += 1

    def word_to_idx(self, word):
        """Convert a word to its index in the vocabulary."""
        return self.word_to_index.get(word, self.word_to_index["<UNK>"])

    def idx_to_word(self, idx):
        """Convert an index to its corresponding word in the vocabulary."""
        return self.index_to_word.get(idx, "<UNK>")


class SwahiliTextDataset(Dataset):
    """PyTorch Dataset for Swahili text."""

    def __init__(self, text, vocab, sequence_length):
        self.text = text
        self.vocab = vocab
        self.sequence_length = sequence_length
        self.preprocessor = SwahiliPreprocessor()

        # Clean and tokenize text
        cleaned_text = self.preprocessor.clean_text(text)
        self.words = self.preprocessor.tokenize(cleaned_text)

        # Convert words to indices
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


class SwahiliLSTMModel(nn.Module):
    """LSTM-based language model for Swahili."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(SwahiliLSTMModel, self).__init__()

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


def load_and_preprocess_data(file_path):
    """Load and preprocess Swahili data from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return ""


def build_vocabulary(training_text):
    """Build vocabulary from training text."""
    preprocessor = SwahiliPreprocessor()
    vocab = SwahiliVocabulary()

    # Clean and tokenize text
    cleaned_text = preprocessor.clean_text(training_text)
    words = preprocessor.tokenize(cleaned_text)

    # Count word frequencies
    for word in words:
        vocab.add_word(word)

    # Build vocabulary from counts
    vocab.build_vocab(min_freq=MIN_WORD_FREQ)

    return vocab


def train_model(model, train_loader, valid_loader, epochs, learning_rate, device, model_path):
    """Train the Swahili language model."""
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # Track best validation loss
    best_valid_loss = float('inf')

    # Training loop
    for epoch in range(epochs):
        start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs, _ = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            # Apply gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            # Accumulate loss
            train_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        valid_loss = 0

        with torch.no_grad():
            for inputs, targets in tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{epochs} [Valid]"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()

        # Calculate average validation loss
        avg_valid_loss = valid_loss / len(valid_loader)

        # Update learning rate scheduler
        scheduler.step(avg_valid_loss)

        # Save best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'valid_loss': avg_valid_loss,
            }, model_path)
            print(f"Model saved to {model_path}")

        # Print epoch statistics
        elapsed_time = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{epochs} - Time: {elapsed_time:.2f}s - Train Loss: {avg_train_loss:.4f} - Valid Loss: {avg_valid_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")

    return model


def generate_swahili_text(model, vocab, seed_text, sequence_length, max_length=100, temperature=0.8, device=device):
    """Generate Swahili text using the trained model."""
    model.eval()  # Set to evaluation mode

    # Preprocess seed text
    preprocessor = SwahiliPreprocessor()
    cleaned_seed = preprocessor.clean_text(seed_text)
    seed_words = preprocessor.tokenize(cleaned_seed)

    # Convert words to indices
    current_indices = [vocab.word_to_idx(word) for word in seed_words]

    # Pad if sequence is shorter than required
    if len(current_indices) < sequence_length:
        padding = [vocab.word_to_idx("<PAD>")] * (sequence_length - len(current_indices))
        current_indices = padding + current_indices

    # Truncate if sequence is longer than required
    if len(current_indices) > sequence_length:
        current_indices = current_indices[-sequence_length:]

    # Start with seed text
    generated_words = seed_words.copy()

    # Generate new words
    for _ in range(max_length):
        # Convert to tensor and add batch dimension
        x = torch.tensor([current_indices], dtype=torch.long).to(device)

        # Generate prediction
        with torch.no_grad():
            output, _ = model(x)

        # Apply temperature to output logits
        output = output.squeeze() / temperature
        probs = torch.softmax(output, dim=0).cpu().numpy()

        # Sample from the distribution
        next_index = np.random.choice(len(probs), p=probs)
        next_word = vocab.idx_to_word(next_index)

        # Skip padding and special tokens
        if next_word in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
            continue

        # Add new word to generated text
        generated_words.append(next_word)

        # Update current sequence
        current_indices = current_indices[1:] + [next_index]

        # End if generated text is long enough or we hit an end token
        if next_word in [".", "?", "!"] and len(generated_words) > max_length // 2:
            if random.random() < 0.5:  # 50% chance to end the sentence
                break

    # Join words to form sentence
    generated_text = " ".join(generated_words)

    # Capitalize first letter
    if generated_text:
        generated_text = generated_text[0].upper() + generated_text[1:]

    return generated_text


def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    perplexity = np.exp(avg_test_loss)

    print(f"Test Loss: {avg_test_loss:.4f} - Perplexity: {perplexity:.4f}")

    return avg_test_loss, perplexity


def main():
    """Main function to train Swahili language model."""
    parser = argparse.ArgumentParser(description='Swahili Language Model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--generate', action='store_true', help='Generate text')
    parser.add_argument('--seed_text', type=str, default='habari', help='Seed text for generation')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for generation')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length for generation')

    args = parser.parse_args()

    # Set file paths
    training_file = os.path.join(SWAHILI_DATA_DIR, 'train.txt')
    validation_file = os.path.join(SWAHILI_DATA_DIR, 'valid.txt')
    test_file = os.path.join(SWAHILI_DATA_DIR, 'test.txt')
    model_path = os.path.join(MODELS_DIR, 'swahili_language_model.pt')
    vocab_path = os.path.join(MODELS_DIR, 'swahili_vocab.pkl')

    if args.train:
        print("Loading training data...")
        training_text = load_and_preprocess_data(training_file)
        validation_text = load_and_preprocess_data(validation_file)

        # Build vocabulary from training data
        print("Building vocabulary...")
        vocab = build_vocabulary(training_text)
        print(f"Vocabulary size: {vocab.n_words} words")

        # Save vocabulary
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"Vocabulary saved to {vocab_path}")

        # Create datasets and dataloaders
        train_dataset = SwahiliTextDataset(training_text, vocab, SEQUENCE_LENGTH)
        valid_dataset = SwahiliTextDataset(validation_text, vocab, SEQUENCE_LENGTH)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

        print(f"Training data: {len(train_dataset)} sequences")
        print(f"Validation data: {len(valid_dataset)} sequences")

        # Create model
        model = SwahiliLSTMModel(
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
            train_loader=train_loader,
            valid_loader=valid_loader,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            device=device,
            model_path=model_path
        )

        print("Training complete!")

    if args.evaluate:
        # Load test data
        print("Loading test data...")
        test_text = load_and_preprocess_data(test_file)

        # Load vocabulary
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        # Create test dataset and dataloader
        test_dataset = SwahiliTextDataset(test_text, vocab, SEQUENCE_LENGTH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Load model
        model = SwahiliLSTMModel(
            vocab_size=vocab.n_words,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        ).to(device)

        # Load saved model state
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate model
        print("\nEvaluating model on test data...")
        test_loss, perplexity = evaluate_model(model, test_loader, device)

        print(f"Test Results - Loss: {test_loss:.4f} - Perplexity: {perplexity:.4f}")

    if args.generate:
        # Load vocabulary
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        # Load model
        model = SwahiliLSTMModel(
            vocab_size=vocab.n_words,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        ).to(device)

        # Load saved model state
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Generate text
        print(f"\nGenerating Swahili text starting with: '{args.seed_text}'")

        for i in range(5):  # Generate 5 different texts
            generated_text = generate_swahili_text(
                model=model,
                vocab=vocab,
                seed_text=args.seed_text,
                sequence_length=SEQUENCE_LENGTH,
                max_length=args.max_length,
                temperature=args.temperature,
                device=device
            )

            print(f"\nGenerated text {i + 1}:")
            print(generated_text)

    # If no options specified, print help
    if not (args.train or args.evaluate or args.generate):
        parser.print_help()


if __name__ == "__main__":
    main()
