"""Generate text with a trained microgpt snapshot."""

import argparse
import random

from dataset import encode
from model import softmax
from snapshot import load_snapshot


def generate(model, vocabulary, prompt, temperature):
    if temperature <= 0:
        raise ValueError('temperature must be greater than 0')
    if len(prompt) > model.config.block_size:
        raise ValueError(
            f"prompt is longer than the model's block size "
            f'({model.config.block_size})'
        )

    prompt_tokens = encode(prompt, vocabulary)
    # For the checked-in vocabulary, prompt="Sch" maps to [15, 22, 27]. These
    # known IDs are fed through the same embedding table used during training.
    bos = len(vocabulary)
    # Generation begins from BOS just as every training example did. The cache
    # persists for this sample so each new character can attend to its prefix.
    keys, values = model.new_cache()
    token_id = bos
    output = []

    for position_id in range(model.config.block_size):
        # forward(token_id) predicts the token *after* token_id. At position 0,
        # BOS therefore predicts the first character of a new name.
        logits = model.forward(token_id, position_id, keys, values)
        if position_id < len(prompt_tokens):
            # "Teacher force" the supplied prompt: for prompt="Sch", consume S,
            # c, and h instead of sampling, while still adding them to the cache.
            # The exact alignment is:
            #   position 0: input BOS, choose forced S
            #   position 1: input S,   choose forced c
            #   position 2: input c,   choose forced h
            #   position 3: input h,   sample the first new character
            token_id = prompt_tokens[position_id]
        else:
            # Temperature reshapes uncertainty. Values below 1 sharpen preferences;
            # values above 1 flatten them. Sampling (rather than argmax) lets one
            # snapshot produce varied names across repeated calls.
            # Dividing logits [2, 1] by temperature 0.5 gives [4, 2], making the
            # larger score much more likely; temperature 2 gives [1, 0.5], making
            # the distribution more even.
            probabilities = softmax([value / temperature for value in logits])
            # random.choices draws one token ID using those probabilities as
            # weights. The model may choose any vocabulary character or BOS/end.
            token_id = random.choices(
                range(model.vocab_size),
                weights=[probability.data for probability in probabilities],
            )[0]

        if token_id == bos:
            # During training BOS followed every name, so sampling it now means
            # the model considers the generated name complete.
            break
        # BOS was handled above, so token_id is now guaranteed to index a printable
        # vocabulary character. Joining these characters later reconstructs text.
        output.append(vocabulary[token_id])

    return ''.join(output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', default='microgpt-snapshot.json')
    parser.add_argument('--prompt', default='S')
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument(
        '--seed',
        type=int,
        help='random seed for reproducible output (system randomness by default)',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.samples < 1:
        raise ValueError('samples must be at least 1')

    if args.seed is not None:
        random.seed(args.seed)
    model, vocabulary = load_snapshot(args.snapshot)
    print(f'vocab size: {model.vocab_size}')
    print(f'num params: {len(model.parameters)}')
    print('\n--- inference (new, hallucinated names) ---')
    for sample_index in range(args.samples):
        sample = generate(model, vocabulary, args.prompt, args.temperature)
        print(f'sample {sample_index + 1:2d}: {sample}')


if __name__ == '__main__':
    main()
