import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"]
    words = word_tokenize(text)

    # Parameters for controlling the intensity of the transformation
    # Increased probabilities to achieve >4 point accuracy drop
    synonym_replacement_prob = 0.20
    typo_prob = 0.18

    # QWERTY keyboard layout for realistic typos
    qwerty_keyboard = {
        'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wsdfr',
        'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujnb', 'i': 'ujklo', 'j': 'huiknm',
        'k': 'jiolm', 'l': 'kopm', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
        'p': 'ol', 'q': 'wa', 'r': 'edf', 's': 'awedxz', 't': 'rfgy',
        'u': 'yihj', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu', 'z': 'asx'
    }

    # Synonym Replacement
    transformed_words = []
    for word in words:
        if word.isalpha() and random.random() < synonym_replacement_prob:
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name() != word:
                        synonyms.append(lemma.name().replace('_', ' '))
            if synonyms:
                transformed_words.append(random.choice(synonyms))
            else:
                transformed_words.append(word)
        else:
            transformed_words.append(word)

    # Character-Level Typos
    final_words = []
    for word in transformed_words:
        # Apply typos to words with length > 2 (instead of > 3) to affect more words
        if word.isalpha() and len(word) > 2 and random.random() < typo_prob:
            typo_type = random.choice(['swap', 'delete', 'insert', 'substitute'])

            if typo_type == 'swap' and len(word) > 1:
                pos = random.randint(0, len(word) - 2)
                word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
            elif typo_type == 'delete' and len(word) > 1:
                pos = random.randint(0, len(word) - 1)
                word = word[:pos] + word[pos+1:]
            elif typo_type == 'insert':
                pos = random.randint(0, len(word))
                if word and pos > 0 and word[pos-1].lower() in qwerty_keyboard:
                    char_to_insert = random.choice(qwerty_keyboard[word[pos-1].lower()])
                else:
                    char_to_insert = random.choice('abcdefghijklmnopqrstuvwxyz')
                word = word[:pos] + char_to_insert + word[pos:]
            elif typo_type == 'substitute':
                pos = random.randint(0, len(word) - 1)
                if word[pos].lower() in qwerty_keyboard:
                    char_to_substitute = random.choice(qwerty_keyboard[word[pos].lower()])
                else:
                    char_to_substitute = random.choice('abcdefghijklmnopqrstuvwxyz')
                word = word[:pos] + char_to_substitute + word[pos+1:]
        final_words.append(word)

    # Detokenize back to sentence
    detokenizer = TreebankWordDetokenizer()
    example["text"] = detokenizer.detokenize(final_words)

    ##### YOUR CODE ENDS HERE ######

    return example


def show_transformed_examples(num_examples=5):
    """
    Shows examples of the custom_transform function on the training data.
    """
    dataset = load_dataset("imdb")
    small_dataset = dataset["train"].shuffle(seed=42).select(range(num_examples))
    
    print(f"Showing {num_examples} transformed examples from the training set:")
    print("=" * 50)

    for i in range(num_examples):
        original_example = small_dataset[i]
        transformed_example = custom_transform(original_example.copy())

        print(f"--- Example {i+1} ---")
        print("Original:")
        print(original_example["text"])
        print("\nTransformed:")
        print(transformed_example["text"])
        print("=" * 50)

if __name__ == '__main__':
    # This allows you to run this script directly to see the transformations
    # e.g., python utils.py
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    show_transformed_examples(num_examples=10)
