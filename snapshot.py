"""Save and load portable JSON snapshots of trained microgpt models."""

import json
from dataclasses import asdict

from model import MicroGPT, ModelConfig

FORMAT_VERSION = 1


def save_snapshot(path, model, vocabulary):
    # JSON stores ordinary floats, not Value objects or their temporary autograd
    # graphs. It contains everything inference needs: token order, dimensions,
    # and learned matrices (for example, wte has [vocab_size, n_embd] entries).
    snapshot = {
        # Allows a future loader to reject or migrate incompatible file layouts.
        'format_version': FORMAT_VERSION,
        # Token IDs are list positions: uchars[10] is the character for token 10.
        # BOS/end is implicit at index len(uchars), so it is not stored here.
        'uchars': vocabulary,
        # Example: {'n_layer': 1, 'n_embd': 16, 'block_size': 16, 'n_head': 4}.
        'config': asdict(model.config),
        # Dict of named 2-D float lists such as wte and layer0.attn_wq.
        'state_dict': model.export_weights(),
    }
    # ensure_ascii=False keeps names such as "Schäfer" readable instead of
    # escaping their vocabulary characters as Unicode code points.
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(snapshot, file, ensure_ascii=False)


def load_snapshot(path):
    # Loading is the boundary between persistence and computation: JSON supplies
    # plain lists/floats, then MicroGPT wraps each weight in a Value object again.
    with open(path, encoding='utf-8') as file:
        snapshot = json.load(file)

    if snapshot.get('format_version') != FORMAT_VERSION:
        raise ValueError(f'unsupported snapshot format in {path!r}')

    vocabulary = snapshot['uchars']
    # Reconstructing the exact architecture is essential because stored matrices
    # only make sense with the embedding width, layer count, and context length
    # under which they were trained.
    config = ModelConfig(**snapshot['config'])
    model = MicroGPT(
        vocab_size=len(vocabulary) + 1,
        config=config,
        weight_data=snapshot['state_dict'],
    )
    return model, vocabulary
