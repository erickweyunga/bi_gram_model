import os
import random
import sys
import torch
import pickle
import readline  # Enables arrow key navigation and history in input
import argparse
import numpy as np
import re
import colorama
from colorama import Fore, Style

# Import the model and necessary functions
from main import SwahiliLSTMModel, SwahiliPreprocessor, generate_swahili_text

# Initialize colorama for colored terminal output
colorama.init()

MODELS_DIR = "./swahili_models"
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "swahili_language_model.pt")
DEFAULT_VOCAB_PATH = os.path.join(MODELS_DIR, "swahili_vocab.pkl")

# Configuration
EMBEDDING_DIM = 200
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
SEQUENCE_LENGTH = 20
MAX_RESPONSE_WORDS = 50

# Set device for inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model_and_vocab(model_path, vocab_path):
    """Load the trained Swahili model and vocabulary."""
    print(f"{Fore.YELLOW}Loading Swahili model from {model_path}...{Style.RESET_ALL}")

    # Load vocabulary
    try:
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"{Fore.GREEN}Vocabulary loaded successfully with {vocab.n_words} words{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error loading vocabulary: {e}{Style.RESET_ALL}")
        sys.exit(1)

    # Create model with the same architecture
    model = SwahiliLSTMModel(
        vocab_size=vocab.n_words,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    # Load model weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode
        print(f"{Fore.GREEN}Model loaded successfully{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error loading model: {e}{Style.RESET_ALL}")
        sys.exit(1)

    return model, vocab


def enhance_swahili_input(user_input):
    """Enhance user input for better generation in Swahili."""
    # Common Swahili words and phrases to expand context
    enhancements = {
        "habari": "habari ya leo ni",
        "jambo": "jambo kwa watu wote",
        "karibu": "karibu kwetu",
        "asante": "asante sana kwa",
        "ndiyo": "ndiyo hii ni",
        "hapana": "hapana sikufikiria",
        "sasa": "sasa hivi tuna",
        "kwaheri": "kwaheri na",
    }

    # For single words, try to enhance
    if user_input.strip().lower() in enhancements:
        return enhancements[user_input.strip().lower()]

    return user_input


def generate_response(model, vocab, user_input, temperature=0.8, attempts=3):
    """Generate a response based on user input."""
    best_response = None
    best_length = 0

    # Try multiple times to get a good response
    for _ in range(attempts):
        # Enhance input for better context
        enhanced_input = enhance_swahili_input(user_input)

        # Generate response
        response = generate_swahili_text(
            model=model,
            vocab=vocab,
            seed_text=enhanced_input,
            sequence_length=SEQUENCE_LENGTH,
            max_length=MAX_RESPONSE_WORDS,
            temperature=temperature,
            device=device
        )

        # Remove the seed text from the response
        if response.lower().startswith(enhanced_input.lower()):
            response = response[len(enhanced_input):].strip()

        # Check if this response is better than previous ones
        if response and len(response.split()) > best_length:
            best_response = response
            best_length = len(response.split())

    # If no good response, return a fallback
    if not best_response or best_length < 3:
        fallbacks = [
            "Samahani, sielewi vizuri. Unaweza kuuliza tena?",
            "Sikufahamu. Tafadhali jaribu tena.",
            "Hmm, ni swali ngumu. Hebu tujaribu mada nyingine."
        ]
        return random.choice(fallbacks)

    return best_response


def clean_response(response):
    """Clean up the generated response for better readability."""
    # Remove leading punctuation
    response = re.sub(r'^[,.;:!?]', '', response).strip()

    # Ensure proper capitalization
    if response and response[0].islower():
        response = response[0].upper() + response[1:]

    # Ensure proper ending
    if response and not any(response.endswith(end) for end in ['.', '?', '!']):
        response += '.'

    return response


def print_banner():
    """Print a welcome banner for the Swahili chatbot."""
    banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   {Fore.YELLOW}Swahili AI Chatbot - Faragha ya Kweli{Fore.CYAN}                 ║
║   Powered by Neural Language Model                       ║
║                                                          ║
║   Type {Fore.GREEN}'msaada'{Fore.CYAN} for help or {Fore.GREEN}'toka'{Fore.CYAN} to quit             ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner)


def print_help():
    """Print help information in Swahili."""
    help_text = f"""
{Fore.YELLOW}Amri Zinazopatikana:{Style.RESET_ALL}
  {Fore.GREEN}msaada{Style.RESET_ALL}           - Onyesha ujumbe huu wa msaada
  {Fore.GREEN}toka{Style.RESET_ALL}, {Fore.GREEN}ondoka{Style.RESET_ALL}   - Toka kwenye chatbot
  {Fore.GREEN}joto=<thamani>{Style.RESET_ALL}  - Weka joto (0.1-2.0) la uzalishaji wa maandishi
                    Chini = zaidi ya kulenga, Juu = zaidi ya nasibu
  {Fore.GREEN}safi{Style.RESET_ALL}             - Safisha skrini

{Fore.YELLOW}Vidokezo:{Style.RESET_ALL}
  - Anza na maneno ya Kiswahili kama "habari", "jambo", "karibu", etc.
  - Jaribu thamani tofauti za joto ili kubadilisha ubunifu
  - Tumia sentensi fupi kwa matokeo bora
"""
    print(help_text)


def main():
    """Main function to run the Swahili chatbot CLI."""
    import random

    parser = argparse.ArgumentParser(description='Swahili AI Chatbot')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to the trained model')
    parser.add_argument('--vocab', type=str, default=DEFAULT_VOCAB_PATH,
                        help='Path to the vocabulary file')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for text generation (0.1-2.0)')

    args = parser.parse_args()

    # Load model and vocabulary
    model, vocab = load_model_and_vocab(args.model, args.vocab)

    # Set initial temperature
    temperature = args.temperature

    # Print welcome banner
    print_banner()

    # Main interaction loop
    while True:
        try:
            # Get user input
            user_input = input(f"{Fore.GREEN}Wewe: {Style.RESET_ALL}")

            # Process commands
            if user_input.lower() in ['toka', 'ondoka', 'exit', 'quit']:
                print(f"{Fore.YELLOW}Kwaheri! Tutaonana tena.{Style.RESET_ALL}")
                break
            elif user_input.lower() in ['msaada', 'help']:
                print_help()
                continue
            elif user_input.lower() in ['safi', 'clear']:
                os.system('cls' if os.name == 'nt' else 'clear')
                print_banner()
                continue
            elif user_input.lower().startswith('joto=') or user_input.lower().startswith('temp='):
                try:
                    value_part = user_input.split('=')[1]
                    new_temp = float(value_part)
                    if 0.1 <= new_temp <= 2.0:
                        temperature = new_temp
                        print(f"{Fore.YELLOW}Joto limewekwa kuwa {temperature}{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}Joto lazima liwe kati ya 0.1 na 2.0{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Thamani ya joto si sahihi{Style.RESET_ALL}")
                continue
            elif not user_input.strip():
                continue

            # Generate and print response
            print(f"{Fore.BLUE}Bot: {Style.RESET_ALL}", end="")
            sys.stdout.flush()  # Ensure text is displayed immediately

            response = generate_response(model, vocab, user_input, temperature)
            cleaned_response = clean_response(response)

            print(cleaned_response)

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Unatoka kwenye chatbot...{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"{Fore.RED}Hitilafu: {e}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
