"""Train microgpt and write an inference snapshot."""

import argparse
import random

from dataset import build_vocabulary, download_if_missing, encode, read_documents
from model import MicroGPT, ModelConfig, softmax
from snapshot import save_snapshot


def train(model: MicroGPT, documents, vocabulary, steps):
    if steps < 1:
        raise ValueError('steps must be at least 1')

    # Adam optimizer hyperparameters. learning_rate controls update size; beta1
    # smooths gradients; beta2 smooths squared gradients; epsilon prevents /0.
    learning_rate, beta1, beta2, epsilon = 0.01, 0.85, 0.99, 1e-8
    # Adam keeps two small pieces of history for every trainable scalar. Both
    # begin at zero and persist across documents.
    first_moment = [0.0] * len(model.parameters)
    second_moment = [0.0] * len(model.parameters)
    # BOS uses the one ID not occupied by a real character. Reusing it at both
    # ends teaches the model how names start and when generation should stop.
    bos = len(vocabulary)

    for step in range(steps):
        # This is stochastic training with one document per step, not a batch.
        # Modulo cycles through the shuffled dataset if steps exceeds its size.
        document = documents[step % len(documents)]
        # With the checked-in vocabulary, "Meyer" becomes
        # [BOS, 10, 24, 39, 24, 34, BOS]. Inputs are every item except the last;
        # targets are the same list shifted left by one position.
        tokens = [bos] + encode(document, vocabulary) + [bos]
        # The individual training pairs are therefore BOS->M, M->e, e->y,
        # y->e, e->r, and r->BOS. The last pair teaches when to stop.
        # A 16-position block trains at most 16 next-character predictions. Longer
        # documents are truncated because positional embeddings exist only 0..15.
        sequence_length = min(model.config.block_size, len(tokens) - 1)

        # The cache is fresh for each document but shared across its positions, so
        # position 4 can attend to the representations produced at positions 0..3.
        keys, values = model.new_cache()
        losses = []
        for position_id in range(sequence_length):
            token_id = tokens[position_id]
            target_id = tokens[position_id + 1]
            # By this iteration the cache holds the prefix through token_id.
            # logits[target_id] should eventually become larger than alternatives.
            logits = model.forward(token_id, position_id, keys, values)
            # Example logits are arbitrary scores such as [1.2, -0.4, 2.0];
            # softmax turns them into a probability distribution summing to 1.
            probabilities = softmax(logits)
            # Cross-entropy for the correct next token is -log(p). A confident
            # p=0.9 costs 0.105; a poor p=0.01 costs 4.605.
            losses.append(-probabilities[target_id].log())

        # Averaging makes documents of different lengths contribute comparable
        # loss scales. backward() fills parameter.grad via the full computation.
        loss = sum(losses) / sequence_length
        loss.backward()

        # Linearly decay the learning rate to zero over this run. The following
        # Adam update smooths noisy gradients with first/second moving moments;
        # bias correction matters most during the first few steps.
        current_rate = learning_rate * (1 - step / steps)
        for index, parameter in enumerate(model.parameters):
            # m = beta1*m + (1-beta1)*gradient: an exponentially weighted mean
            # that estimates the gradient's direction while filtering noise.
            first_moment[index] = (
                beta1 * first_moment[index] + (1 - beta1) * parameter.grad
            )
            # v similarly tracks gradient squared, estimating its recent scale.
            # Parameters with consistently large gradients get normalized updates.
            second_moment[index] = (
                beta2 * second_moment[index] + (1 - beta2) * parameter.grad**2
            )
            # Because m and v started at zero, early values are biased downward.
            # Dividing by (1-beta**time) removes that initialization bias.
            corrected_first = first_moment[index] / (1 - beta1 ** (step + 1))
            corrected_second = second_moment[index] / (1 - beta2 ** (step + 1))
            # Gradient descent subtracts the normalized update. A positive
            # corrected_first therefore decreases this parameter's data value.
            parameter.data -= (
                current_rate
                * corrected_first
                / (corrected_second**0.5 + epsilon)
            )
            # Gradients accumulate by default in Value.backward(), so clear each
            # one before the next document builds a new computation graph.
            parameter.grad = 0

        print(f'step {step + 1:4d} / {steps:4d} | loss {loss.data:.4f}', end='\r')

    print()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='input.txt')
    parser.add_argument('--snapshot', default='microgpt-snapshot.json')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    # The seed makes initialization and document shuffling reproducible, which is
    # useful when comparing code changes in an educational example.
    random.seed(args.seed)

    download_if_missing(args.input)
    documents = read_documents(args.input)
    # Without shuffling, consecutive steps would always follow file order. The
    # fixed seed means this random order is nevertheless repeatable.
    random.shuffle(documents)
    vocabulary = build_vocabulary(documents)
    # +1 reserves the BOS/end token described in train(); it deliberately has no
    # printable entry in vocabulary.
    model = MicroGPT(len(vocabulary) + 1, ModelConfig())

    print(f'num docs: {len(documents)}')
    print(f'vocab size: {len(vocabulary) + 1}')
    print(f'num params: {len(model.parameters)}')
    train(model, documents, vocabulary, args.steps)
    save_snapshot(args.snapshot, model, vocabulary)
    print(f'saved snapshot: {args.snapshot}')


if __name__ == '__main__':
    main()
