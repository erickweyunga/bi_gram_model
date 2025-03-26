import heapq
import random
from itertools import islice
from string import punctuation

import nltk
from nltk import bigrams, FreqDist, ConditionalFreqDist

import os
import string

nltk.download('punkt', quiet=True)

input_data_dir = "./data"
topk = 3

punctuation_to_remove = string.punctuation.replace(".", "")


def is_hidden(filepath):
    return os.path.basename(filepath).startswith(".")


def tokenize(text_data):
    words = nltk.word_tokenize(text_data.lower())
    return words


def generate_bigram(words):
    bigram = list(bigrams(words))
    bi_gram_freq_dist = FreqDist(bigram)
    return bi_gram_freq_dist


def conditional_ditribution_bigram(bi_gram_freq_dist):
    bi_gram_freq = ConditionalFreqDist()
    # Populate the conditional frequency distribution correctly
    for (first_word, second_word), freq in bi_gram_freq_dist.items():
        bi_gram_freq[first_word][second_word] = freq
    return bi_gram_freq


def top_k_words(bi_gram_freq, word, k=3):
    """Get the top k words following the given word using direct frequency sorting"""
    if word not in bi_gram_freq:
        return []

    # Get all words that follow the given word, with their frequencies
    following_words = bi_gram_freq[word].items()

    # Sort by frequency in descending order and take the top k
    top_words = sorted(following_words, key=lambda x: x[1], reverse=True)[:k]

    return [(freq, word) for word, freq in top_words]


def build_top_k_dictionary(bi_gram_freq, k=3):
    """Build a dictionary of the top k words for each first word"""
    result = {}

    # For each word in the corpus
    for first_word in bi_gram_freq.keys():
        # Get its top k following words
        result[first_word] = top_k_words(bi_gram_freq, first_word, k)

    return result


def create_weighted_cfd(top_words_dict):
    """Create a conditional frequency distribution from the top words dictionary"""
    cfd = ConditionalFreqDist()

    for first_word, top_words in top_words_dict.items():
        for i, (freq, second_word) in enumerate(top_words):
            # Use actual frequency as the weight
            cfd[first_word][second_word] = freq

    return cfd


def display_top_words(bi_gram_freq, word, n=10):
    """Display the top n words that follow the given word"""
    if word in bi_gram_freq:
        print(f"\nTop {n} words following '{word}':")
        items = bi_gram_freq[word].items()
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)[:n]
        for second_word, freq in sorted_items:
            print(f"  '{second_word}': {freq} occurrences")
    else:
        print(f"'{word}' not found in the corpus")


def generate_sentence(start_word, num_words, bigram_freq):
    """Generate a sentence starting with the given word"""
    current_word = start_word.lower()
    sentence = [current_word]

    for _ in range(num_words):
        if current_word not in bigram_freq or not bigram_freq[current_word]:
            break

        # Get possible next words with their frequencies
        next_word_dict = bigram_freq[current_word]
        words = list(next_word_dict.keys())
        weights = [next_word_dict[word] for word in words]

        if not words:
            break

        # Select next word based on frequency weights
        next_word = random.choices(words, weights=weights, k=1)[0]
        sentence.append(next_word)

        # Move to next word
        current_word = next_word

        # End sentence if we hit a period or similar
        if next_word == ".":
            if len(sentence) > 5 and random.random() < 0.7:  # 70% chance to end after a period if sentence is long enough
                break

    return " ".join(sentence)


def main():
    text_data = ""
    for filename in os.listdir(input_data_dir):
        filepath = os.path.join(input_data_dir, filename)
        if not is_hidden(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        # Remove unwanted punctuation (except periods)
                        for char in punctuation_to_remove:
                            line = line.replace(char, "")
                        text_data += line
    print(f"Total length of text data: {len(text_data)}")

    words = tokenize(text_data)
    bi_gram_freq_dist = generate_bigram(words)

    print(f"BIGram frequency distribution: {bi_gram_freq_dist}")
    track_performance(bi_gram_freq_dist)

    # Get the complete conditional frequency distribution
    bi_gram_freq = conditional_ditribution_bigram(bi_gram_freq_dist)

    # Print detailed info about the word "natural"
    display_top_words(bi_gram_freq, "natural")

    # Build top k dictionary using direct frequency calculation (no heap)
    top_k_dict = build_top_k_dictionary(bi_gram_freq, topk)

    if "natural" in top_k_dict:
        print(f"\nTop {topk} words following 'natural' (direct calculation): {top_k_dict['natural']}")

    # Generate sentence using full frequency information
    print("\nGenerated sentence using all frequencies:")
    full_sentence = generate_sentence("natural", 20, bi_gram_freq)
    print(full_sentence)

    # Generate sentence using top k approach
    top_k_freq = create_weighted_cfd(top_k_dict)
    print(f"\nGenerated sentence using top {topk} approach:")
    top_k_sentence = generate_sentence("natural", 20, top_k_freq)
    print(top_k_sentence)

    # Generate examples from other common words
    common_first_words = ["the", "market", "company", "said", "dollar"]
    print("\nAdditional example sentences:")
    for word in common_first_words:
        if word in bi_gram_freq:
            print(f"\nStarting with '{word}':")
            sentence = generate_sentence(word, 15, bi_gram_freq)
            print(sentence)


def track_performance(bi_gram_freq_dist):
    first_five_items = list(islice(bi_gram_freq_dist.items(), 5))
    for item in first_five_items:
        print(item)
    return first_five_items


if __name__ == "__main__":
    main()