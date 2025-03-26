import os
import sys
import torch
import pickle
import argparse
import re
import colorama
from colorama import Fore, Style

# Import your model definition
from neuro_network.main import LSTMLanguageModel, Vocabulary, generate_text

# Initialize colorama for colored terminal output
colorama.init()

MODELS_DIR = "./models"
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "pytorch_language_model.pt")
DEFAULT_VOCAB_PATH = os.path.join(MODELS_DIR, "pytorch_vocab.pkl")

# Configuration
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2
SEQUENCE_LENGTH = 10
MAX_RESPONSE_WORDS = 50

# Set device for inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model_and_vocab(model_path, vocab_path):
    """Load the trained model and vocabulary."""
    print(f"{Fore.YELLOW}Loading model from {model_path}...{Style.RESET_ALL}")

    # Load vocabulary
    try:
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"{Fore.GREEN}Vocabulary loaded successfully with {vocab.n_words} words{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error loading vocabulary: {e}{Style.RESET_ALL}")
        sys.exit(1)

    # Create model with the same architecture
    model = LSTMLanguageModel(
        vocab_size=vocab.n_words,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    # Load model weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set to evaluation mode
        print(f"{Fore.GREEN}Model loaded successfully{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error loading model: {e}{Style.RESET_ALL}")
        sys.exit(1)

    return model, vocab


def preprocess_input(text):
    """Preprocess user input."""
    # Clean and normalize text
    text = text.lower().strip()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text


def generate_response(model, vocab, user_input, temperature=0.8):
    """Generate a response based on user input."""
    # Preprocess input
    processed_input = preprocess_input(user_input)

    # Generate response
    response = generate_text(
        model=model,
        vocab=vocab,
        seed_text=processed_input,
        sequence_length=SEQUENCE_LENGTH,
        num_words=MAX_RESPONSE_WORDS,
        temperature=temperature,
        device=device
    )

    # Remove the seed text from the response
    if response.lower().startswith(processed_input.lower()):
        response = response[len(processed_input):].strip()

    return response


def print_banner():
    """Print a welcome banner for the chatbot."""
    banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   {Fore.YELLOW}Financial News Chatbot{Fore.CYAN}                               ║
║   Powered by Neural Language Model                       ║
║                                                          ║
║   Type {Fore.GREEN}'help'{Fore.CYAN} for commands or {Fore.GREEN}'exit'{Fore.CYAN} to quit              ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner)


def print_help():
    """Print help information."""
    help_text = f"""
{Fore.YELLOW}Available Commands:{Style.RESET_ALL}
  {Fore.GREEN}help{Style.RESET_ALL}              - Show this help message
  {Fore.GREEN}exit{Style.RESET_ALL}, {Fore.GREEN}quit{Style.RESET_ALL}       - Exit the chatbot
  {Fore.GREEN}temp=<value>{Style.RESET_ALL}      - Set temperature (0.1-2.0) for text generation
                    Lower = more focused, Higher = more random
  {Fore.GREEN}clear{Style.RESET_ALL}             - Clear the screen

{Fore.YELLOW}Tips:{Style.RESET_ALL}
  - Start with financial terms like "the market", "stocks", "dollar", etc.
  - Try different temperature values to adjust creativity
  - Responses are generated based on financial news patterns
"""
    print(help_text)


def main():
    """Main function to run the chatbot CLI."""
    parser = argparse.ArgumentParser(description='Financial News Chatbot')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to the trained model')
    parser.add_argument('--vocab', type=str, default=DEFAULT_VOCAB_PATH,
                        help='Path to the vocabulary file')
    parser.add_argument('--temperature', type=float, default=0.8,
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
            user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}")

            # Process commands
            if user_input.lower() in ['exit', 'quit']:
                print(f"{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
                break
            elif user_input.lower() == 'help':
                print_help()
                continue
            elif user_input.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                print_banner()
                continue
            elif user_input.lower().startswith('temp='):
                try:
                    new_temp = float(user_input.split('=')[1])
                    if 0.1 <= new_temp <= 2.0:
                        temperature = new_temp
                        print(f"{Fore.YELLOW}Temperature set to {temperature}{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}Temperature must be between 0.1 and 2.0{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Invalid temperature value{Style.RESET_ALL}")
                continue
            elif not user_input.strip():
                continue

            # Generate and print response
            print(f"{Fore.BLUE}Bot: {Style.RESET_ALL}", end="")
            sys.stdout.flush()  # Ensure text is displayed immediately

            response = generate_response(model, vocab, user_input, temperature)

            if not response:
                response = "I'm not sure how to respond to that. Try asking about financial markets or news."

            print(response)

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Exiting chatbot...{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
