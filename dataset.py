"""Dataset loading and character tokenization helpers."""

import os


DEFAULT_NAMES_URL = (
    'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
)


def download_if_missing(path):
    # Preserve a user-provided dataset. The network is touched only when the
    # requested input file does not already exist.
    if os.path.exists(path):
        return

    # Import lazily because normal training with the checked-in input.txt does
    # not need urllib at all.
    import urllib.request
    urllib.request.urlretrieve(DEFAULT_NAMES_URL, path)


def read_documents(path):
    """Read non-empty lines as UTF-8, falling back to Latin-1."""
    # One line is one independent training document/name. For a file containing
    # "Meyer\n\nSchulz\n", the result is ['Meyer', 'Schulz']; blank lines and
    # surrounding whitespace carry no training signal and are discarded.
    for encoding in ('utf-8', 'latin-1'):
        try:
            with open(path, encoding=encoding) as file:
                documents = [line.strip() for line in file if line.strip()]
            if not documents:
                raise ValueError(f'no documents found in {path!r}')
            return documents
        except UnicodeDecodeError:
            # German names may arrive in legacy Latin-1. Retry from the beginning
            # so characters such as ä, ö, and ü remain individual characters.
            pass
    raise UnicodeError(f'could not decode {path!r} as UTF-8 or Latin-1')


def build_vocabulary(documents):
    # Character-level tokenization gives every distinct character one integer ID.
    # For ['Ada', 'Ava'], this returns ['A', 'a', 'd', 'v']; the model later adds
    # one extra ID, len(vocabulary), to mean both "begin" and "end of document".
    # Sorting makes IDs deterministic: the same characters produce the same
    # character-to-ID mapping regardless of document order or set iteration.
    return sorted(set(''.join(documents)))


def encode(text, vocabulary):
    # Example: vocabulary=['A', 'a', 'd', 'v'] turns 'Ada' into [0, 2, 1].
    # Unlike a word tokenizer, each input character therefore consumes one
    # position in the model's context window.
    token_by_character = {character: index for index, character in enumerate(vocabulary)}
    # Fail explicitly rather than silently mapping an unseen character. There is
    # no UNK token in this intentionally tiny tokenizer.
    unknown = sorted(set(text) - token_by_character.keys())
    if unknown:
        raise ValueError(f'text contains characters outside the vocabulary: {unknown}')
    return [token_by_character[character] for character in text]
