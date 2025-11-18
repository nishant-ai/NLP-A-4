#!/usr/bin/env python3
"""
Data Augmentation for Text-to-SQL Training
Generates additional training examples through:
1. Synonym replacement for common words
2. Phrase variations
3. Word order changes where grammatically valid
"""

import os
import random
import re

# Synonym mappings for common words in flight queries
SYNONYMS = {
    # Action words
    'show': ['list', 'display', 'give me', 'find', 'get'],
    'list': ['show', 'display', 'give me', 'find'],
    'find': ['show', 'list', 'get', 'give me'],
    'give me': ['show', 'list', 'find'],
    'get': ['show', 'find', 'list'],

    # Flight-related
    'flights': ['flight', 'planes', 'aircraft'],
    'flight': ['flights'],

    # Location words
    'from': ['leaving', 'departing from', 'out of'],
    'to': ['arriving at', 'going to', 'into'],
    'leaving': ['departing', 'from'],
    'arriving': ['landing', 'getting to'],

    # Time words
    'morning': ['am', 'before noon'],
    'afternoon': ['pm', 'after noon'],
    'evening': ['night', 'late'],

    # Quantity words
    'all': ['every', 'all the'],
    'any': ['some', 'any available'],

    # Question starters
    'what': ['which'],
    'which': ['what'],
    'are there': ['do you have', 'is there'],
    'do you have': ['are there', 'is there'],
    'can you': ['could you', 'would you'],

    # Misc
    'nonstop': ['direct', 'non-stop'],
    'direct': ['nonstop', 'non-stop'],
    'round trip': ['roundtrip', 'return'],
    'one way': ['oneway', 'single'],
    'cheapest': ['least expensive', 'lowest cost'],
    'expensive': ['costly', 'high priced'],
}

# Phrase variations
PHRASE_VARIATIONS = [
    # "flights from X to Y" variations
    (r'flights from (\w+) to (\w+)', [
        r'flights from \1 to \2',
        r'flights to \2 from \1',
        r'\1 to \2 flights',
    ]),
    # "show me" variations
    (r'^show me ', [
        'show me ',
        'list ',
        'find ',
        'give me ',
        'i need ',
    ]),
    # "i want" variations
    (r"^i want ", [
        'i want ',
        'i need ',
        "i'd like ",
        'i would like ',
    ]),
    # "i need" variations
    (r"^i need ", [
        'i need ',
        'i want ',
        "i'd like ",
        'please show me ',
    ]),
]


def synonym_replace(text, prob=0.3):
    """
    Replace words with synonyms randomly.

    Args:
        text: Input natural language query
        prob: Probability of replacing each word

    Returns:
        Augmented text
    """
    words = text.lower().split()
    new_words = []

    i = 0
    while i < len(words):
        # Check for multi-word phrases first
        replaced = False
        for phrase_len in [3, 2]:  # Check 3-word then 2-word phrases
            if i + phrase_len <= len(words):
                phrase = ' '.join(words[i:i+phrase_len])
                if phrase in SYNONYMS and random.random() < prob:
                    replacement = random.choice(SYNONYMS[phrase])
                    new_words.append(replacement)
                    i += phrase_len
                    replaced = True
                    break

        if not replaced:
            word = words[i]
            if word in SYNONYMS and random.random() < prob:
                new_words.append(random.choice(SYNONYMS[word]))
            else:
                new_words.append(word)
            i += 1

    return ' '.join(new_words)


def phrase_variation(text):
    """
    Apply phrase-level variations.

    Args:
        text: Input natural language query

    Returns:
        Augmented text or original if no pattern matches
    """
    text_lower = text.lower()

    for pattern, variations in PHRASE_VARIATIONS:
        if re.search(pattern, text_lower):
            # Choose a random variation
            variation = random.choice(variations)
            return re.sub(pattern, variation, text_lower, count=1)

    return text


def augment_query(nl_query, num_augments=2):
    """
    Generate augmented versions of a natural language query.

    Args:
        nl_query: Original natural language query
        num_augments: Number of augmented versions to generate

    Returns:
        List of augmented queries (may include duplicates of original)
    """
    augmented = []

    for _ in range(num_augments):
        # Apply different augmentation strategies
        strategy = random.choice(['synonym', 'phrase', 'both'])

        if strategy == 'synonym':
            aug = synonym_replace(nl_query, prob=0.4)
        elif strategy == 'phrase':
            aug = phrase_variation(nl_query)
        else:  # both
            aug = synonym_replace(nl_query, prob=0.3)
            aug = phrase_variation(aug)

        # Only add if different from original
        if aug.lower() != nl_query.lower():
            augmented.append(aug)

    return augmented


def augment_dataset(nl_file, sql_file, output_nl_file, output_sql_file, augment_factor=2):
    """
    Augment the entire training dataset.

    Args:
        nl_file: Path to original natural language queries
        sql_file: Path to original SQL queries
        output_nl_file: Path to save augmented NL queries
        output_sql_file: Path to save augmented SQL queries
        augment_factor: How many augmented versions per original
    """
    # Load original data
    with open(nl_file, 'r') as f:
        nl_queries = [line.strip() for line in f.readlines()]

    with open(sql_file, 'r') as f:
        sql_queries = [line.strip() for line in f.readlines()]

    assert len(nl_queries) == len(sql_queries), "NL and SQL files must have same number of lines"

    # Generate augmented data
    augmented_nl = []
    augmented_sql = []

    # Keep original data
    augmented_nl.extend(nl_queries)
    augmented_sql.extend(sql_queries)

    # Add augmented versions
    for nl, sql in zip(nl_queries, sql_queries):
        augs = augment_query(nl, num_augments=augment_factor)
        for aug_nl in augs:
            augmented_nl.append(aug_nl)
            augmented_sql.append(sql)  # SQL stays the same

    # Shuffle the augmented data
    combined = list(zip(augmented_nl, augmented_sql))
    random.shuffle(combined)
    augmented_nl, augmented_sql = zip(*combined)

    # Save augmented data
    with open(output_nl_file, 'w') as f:
        for query in augmented_nl:
            f.write(query + '\n')

    with open(output_sql_file, 'w') as f:
        for query in augmented_sql:
            f.write(query + '\n')

    print(f"Original dataset: {len(nl_queries)} examples")
    print(f"Augmented dataset: {len(augmented_nl)} examples")
    print(f"Saved to: {output_nl_file} and {output_sql_file}")

    return len(augmented_nl)


def main():
    """Main function to run data augmentation."""
    # Set random seed for reproducibility
    random.seed(42)

    # Paths
    data_dir = 'data'

    # Input files
    train_nl = os.path.join(data_dir, 'train.nl')
    train_sql = os.path.join(data_dir, 'train.sql')

    # Output files (augmented)
    aug_nl = os.path.join(data_dir, 'train_augmented.nl')
    aug_sql = os.path.join(data_dir, 'train_augmented.sql')

    # Run augmentation
    print("=" * 50)
    print("Data Augmentation for Text-to-SQL")
    print("=" * 50)

    num_examples = augment_dataset(
        train_nl, train_sql,
        aug_nl, aug_sql,
        augment_factor=2  # Generate 2 augmented versions per original
    )

    print("\n" + "=" * 50)
    print("Augmentation complete!")
    print("=" * 50)

    # Show some examples
    print("\nSample augmented queries:")
    with open(aug_nl, 'r') as f:
        lines = f.readlines()
        for i in range(min(5, len(lines))):
            print(f"  {i+1}. {lines[i].strip()}")


if __name__ == "__main__":
    main()
