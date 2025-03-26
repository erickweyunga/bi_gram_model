import os
import random
import string
import pickle
import re
import nltk
import math
from nltk import word_tokenize, sent_tokenize
from nltk.util import ngrams
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Any, Optional

# Download required NLTK resources
nltk.download('punkt', quiet=True)

# Configuration
INPUT_DATA_DIR = "../data"
MODELS_DIR = "./models"
OUTPUT_DIR = "./output"
NGRAM_ORDER = 5  # Increased to 5 grams for more context
RANDOM_SEED = 42
SAMPLE_SIZE = 500000  # Significantly increased sample size

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set random seed for reproducibility
random.seed(RANDOM_SEED)


def is_hidden(filepath):
    """Check if file is hidden."""
    return os.path.basename(filepath).startswith(".")


def clean_text(text):
    """Clean the text while preserving sentence structure and key formatting."""
    # Replace line breaks and excessive whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Keep only letters, numbers, spaces, and basic punctuation
    keep_chars = ".?!,;:()'\""  # Extended to include more meaningful punctuation
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

    # Fix spaces after quotes
    cleaned_text = re.sub(r'"\s+', '" ', cleaned_text)

    # Remove excess spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text


def extract_sentences(text):
    """Extract well-formed sentences from text with improved sentence boundary detection."""
    # Clean the text first
    cleaned_text = clean_text(text)

    # Try to extract complete sentences
    sentences = sent_tokenize(cleaned_text)
    result = []

    for sentence in sentences:
        # Further refinement of the sentence
        sentence = sentence.strip()
        if sentence and len(sentence.split()) >= 3:  # Only keep sentences with at least 3 words
            # Check if the sentence has balanced quotes and parentheses
            if sentence.count('"') % 2 == 0 and sentence.count('(') == sentence.count(')'):
                result.append(sentence.lower())

    return result


def load_sample_data():
    """Load a representative sample of text data from files."""
    all_sentences = []
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
                        all_sentences.extend(sentences)

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

    print(f"Finished loading {files_processed} files. Found {len(all_sentences)} sentences.")
    return all_sentences


def build_hierarchical_ngram_model(sentences, max_order=5):
    """Build a hierarchical n-gram model with fallback capabilities."""
    print(f"Building hierarchical n-gram model up to order {max_order}...")

    # Create separate models for each n-gram order
    models = {}
    for n in range(1, max_order + 1):
        print(f"  Processing {n}-grams...")
        models[n] = defaultdict(Counter)

    # Process each sentence
    for sentence in sentences:
        # Tokenize the sentence
        words = word_tokenize(sentence)

        # Add start and end tokens
        padded_words = ['<s>'] * (max_order - 1) + words + ['</s>']

        # Generate n-grams for all orders and count them
        for n in range(1, max_order + 1):
            for i in range(len(padded_words) - n + 1):
                # For unigrams, just count the current word
                if n == 1:
                    models[n][()][padded_words[i]] += 1
                else:
                    context = tuple(padded_words[i:i + n - 1])
                    next_word = padded_words[i + n - 1]
                    models[n][context][next_word] += 1

    print("Model training complete")
    return models


def compute_good_turing_smoothing(counters):
    """Apply Simple Good-Turing smoothing to improve probability estimates."""
    print("Applying Good-Turing smoothing...")

    # For each context, apply smoothing to the counts
    smoothed_counters = {}

    for context, counter in counters.items():
        # Count how many words occur each number of times
        freq_of_freq = Counter([count for count in counter.values()])

        # Create smoothed counter for this context
        smoothed_counter = Counter()

        for word, count in counter.items():
            # Calculate smoothed count
            if count < 5 and count + 1 in freq_of_freq:
                # Good-Turing smoothing formula
                smoothed_count = (count + 1) * freq_of_freq[count + 1] / freq_of_freq[count]
            else:
                # For higher counts, just use the original count
                smoothed_count = count

            smoothed_counter[word] = smoothed_count

        smoothed_counters[context] = smoothed_counter

    return smoothed_counters


def save_model(model, filepath):
    """Save the trained model to a file."""
    print(f"Saving model to {filepath}...")
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully")


def load_model(filepath):
    """Load a trained model from a file."""
    print(f"Loading model from {filepath}...")
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def get_next_word_hierarchical(models, context_stack, temperature=0.8, avoid_words=None):
    """Get the next word using hierarchical backoff with interpolation."""
    if avoid_words is None:
        avoid_words = set()

    # Start with the highest order available in context_stack
    # and back off to lower orders if necessary
    candidates = []

    # Try each context in the stack, from longest to shortest
    for context in context_stack:
        n = len(context) + 1  # The n-gram order

        if n in models and context in models[n]:
            # Get counter of next words for this context
            next_words = models[n][context]

            # Filter out words to avoid
            filtered_next_words = {word: count for word, count in next_words.items()
                                   if word not in avoid_words}

            if filtered_next_words:
                # Calculate probabilities with temperature scaling
                total = sum(filtered_next_words.values())
                candidates.extend([(word, (count / total) ** (1 / temperature), n)
                                   for word, count in filtered_next_words.items()])

    # If we found candidates at any level, select one based on probabilities
    if candidates:
        # Sort by n-gram order (higher is better) then by probability
        candidates.sort(key=lambda x: (-x[2], -x[1]))

        # Extract words and probabilities for random selection
        words = [candidate[0] for candidate in candidates[:30]]  # Top 30 candidates
        probs = [candidate[1] for candidate in candidates[:30]]

        # Normalize probabilities
        total_prob = sum(probs)
        normalized_probs = [p / total_prob for p in probs]

        # Select based on probabilities
        return random.choices(words, weights=normalized_probs, k=1)[0]

    # If no candidates found at any level, return None
    return None


def prepare_context_stack(text, max_order):
    """Prepare a stack of contexts from a seed text for hierarchical model."""
    # Tokenize seed text
    tokens = word_tokenize(text.lower())

    # Create context stack from longest to shortest
    context_stack = []

    # Add contexts of decreasing order
    for n in range(max_order, 0, -1):
        if len(tokens) >= n - 1:
            context = tuple(tokens[-(n - 1):])
            context_stack.append(context)

    # Add empty context for unigram model
    if not context_stack:
        context_stack.append(())

    return context_stack


def generate_text_hierarchical(models, seed_text, num_words=30, max_order=5, temperature=0.7):
    """Generate text using the hierarchical n-gram model."""
    # Tokenize seed text
    tokens = word_tokenize(seed_text.lower())
    generated_words = tokens.copy()

    # Track recent words to avoid repetition
    recent_words = set(tokens[-5:] if len(tokens) >= 5 else tokens)

    # Generate text
    for _ in range(num_words):
        # Prepare context stack for hierarchical model
        context_stack = prepare_context_stack(seed_text, max_order)

        # Get next word with hierarchical backoff
        next_word = get_next_word_hierarchical(models, context_stack, temperature, avoid_words=recent_words)

        # If no word found, use a common word
        if next_word is None:
            common_words = ["the", "of", "and", "to", "in", "a", "that", "is", "was", "for"]
            next_word = random.choice(common_words)

        # Add word to generated text
        generated_words.append(next_word)
        seed_text = seed_text + " " + next_word

        # Update recent words (keep a window of the last 5 words)
        recent_words.add(next_word)
        if len(recent_words) > 5 and len(generated_words) > 5:
            oldest = generated_words[-6]
            if oldest in recent_words:
                recent_words.remove(oldest)

        # End if we reach a sentence ending or closing mark
        if next_word in ['</s>', '.', '!', '?'] and len(generated_words) > 10:
            # 75% chance to end after a sentence marker if we have reasonable length
            if random.random() < 0.75:
                break

    # Remove any special tokens and join words
    result = ' '.join([w for w in generated_words if w not in ['<s>', '</s>']])

    # Clean up spacing around punctuation
    for char in ".?!,;:)":
        result = re.sub(rf'\s+{re.escape(char)}', char, result)

    # Fix spaces before opening parentheses
    result = re.sub(r'\s+\(', ' (', result)

    # Capitalize first letter
    if result:
        result = result[0].upper() + result[1:]

    return result


def analyze_hierarchical_model(models, word, max_order=5, top_n=10):
    """Analyze the hierarchical n-gram model for a given word."""
    # Tokenize the word
    tokens = word_tokenize(word.lower())

    print(f"\nAnalysis for '{word}':")

    # Prepare context stack
    context_stack = prepare_context_stack(word, max_order)

    # Try each context in the stack, from longest to shortest
    for context in context_stack:
        n = len(context) + 1  # The n-gram order

        if n in models and context in models[n]:
            # Get counter of next words for this context
            next_words = models[n][context]

            if next_words:
                print(f"  {n}-gram context {context}:")

                # Sort by frequency
                sorted_words = sorted(next_words.items(), key=lambda x: x[1], reverse=True)

                total_count = sum(next_words.values())

                for next_word, count in sorted_words[:top_n]:
                    probability = count / total_count
                    print(f"    '{next_word}': count={count}, probability={probability:.6f}")

                break  # Found results at this level, so stop

    # If no context yielded results
    if all(n not in models or context not in models[n] or not models[n][context]
           for context, n in [(ctx, len(ctx) + 1) for ctx in context_stack]):
        print(f"  No continuations found for '{word}' at any n-gram level")


def main():
    """Main function to train or load model and generate text."""
    model_filepath = os.path.join(MODELS_DIR, f"hierarchical_ngram_model_order{NGRAM_ORDER}.pkl")

    # Build new model
    print(f"Building new hierarchical language model (up to {NGRAM_ORDER}-grams)...")

    # Load and process data
    sentences = load_sample_data()

    # Build hierarchical model
    models = build_hierarchical_ngram_model(sentences, NGRAM_ORDER)

    # Apply Good-Turing smoothing to each level
    for n in models:
        models[n] = compute_good_turing_smoothing(models[n])

    save_model(models, model_filepath)

    # Analyze common words
    common_seed_words = ["natural", "the", "market", "company", "said", "dollar", "stock", "bank", "trading"]
    for word in common_seed_words:
        analyze_hierarchical_model(models, word, NGRAM_ORDER)

    # Generate text examples
    print("\nGenerated text examples:")
    for word in common_seed_words:
        for i in range(3):  # Generate 3 examples for each seed
            text = generate_text_hierarchical(models, word, num_words=40, max_order=NGRAM_ORDER)
            print(f"\nStarting with '{word}' (Example {i + 1}):")
            print(text)

            # Save to output file
            output_filepath = os.path.join(OUTPUT_DIR, f"advanced_generated_{word}_{i + 1}.txt")
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(text)


if __name__ == "__main__":
    main()
