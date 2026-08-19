"""The microgpt model, scalar autograd engine, and model math."""

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    # Number of Transformer blocks applied one after another.
    n_layer: int = 1
    # Width of every token representation. Each character becomes 16 learned
    # numbers rather than being represented by its integer token ID directly.
    n_embd: int = 16
    # Maximum number of input positions, and therefore next-token predictions,
    # processed for one document or generated sample.
    block_size: int = 16
    # Attention runs in parallel heads. With n_embd=16 and n_head=4, each head
    # compares four-dimensional query and key vectors.
    n_head: int = 4

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError('n_embd must be divisible by n_head')


class Value:
    """A scalar value and its autograd history.

    Keeping every number scalar makes backpropagation visible: if ``y = x * 3``,
    ``y`` stores ``x`` as a child and 3 as the local derivative dy/dx.
    """

    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        # data is the scalar produced in the forward pass; grad will eventually
        # hold d(loss)/d(this value) after backward().
        self.data = data
        self.grad = 0
        # Each operation remembers its inputs and its derivative with respect to
        # each input. Together, all Values form the computation graph.
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # If out=a+b, then d(out)/da=1 and d(out)/db=1.
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # If out=a*b, its local derivatives are b and a. For a=2, b=3,
        # the returned Value stores data=6 and local_grads=(3, 2).
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        # Power rule: d(x**n)/dx = n*x**(n-1). Only numeric powers are needed
        # here, primarily square, square root, and reciprocal.
        return Value(self.data**other, (self,), (other * self.data**(other - 1),))

    def log(self):
        # Used by cross-entropy: d(log(x))/dx = 1/x.
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        # Used by softmax; exp is its own derivative.
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        # ReLU keeps positive activations and clips negative ones to zero. Its
        # gradient is therefore 1 on the positive side and 0 on the negative side.
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        # An operation can only send its gradient to its inputs after its own
        # gradient is known. The topological ordering guarantees that order.
        topo = []
        visited = set()

        def build_topo(value):
            if value not in visited:
                visited.add(value)
                for child in value._children:
                    build_topo(child)
                topo.append(value)

        build_topo(self)
        # d(loss)/d(loss) = 1 starts the chain rule. For y=x*3 and x=2,
        # calling y.backward() consequently leaves x.grad equal to 3.
        self.grad = 1
        for value in reversed(topo):
            # Chain rule: d(loss)/d(child) += d(value)/d(child) *
            # d(loss)/d(value). += is necessary when a child feeds several paths.
            for child, local_grad in zip(value._children, value._local_grads):
                child.grad += local_grad * value.grad


def linear(x, weights):
    # Matrix-vector multiplication. With x shaped [16] and weights [64, 16],
    # the returned vector has 64 Values (the MLP expansion used below).
    return [sum(weight * value for weight, value in zip(row, x)) for row in weights]


def softmax(logits):
    # Convert arbitrary scores into probabilities that sum to 1. Subtracting
    # max(logits) is numerically safer and changes no result: softmax([1, 2])
    # is approximately [0.269, 0.731], just like softmax([-1, 0]).
    max_value = max(value.data for value in logits)
    exps = [(value - max_value).exp() for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def rmsnorm(x):
    # Normalize the vector's overall magnitude while preserving its direction.
    # For x=[3, 4], mean_square=12.5 and scale is about 0.283.
    mean_square = sum(value * value for value in x) / len(x)
    scale = (mean_square + 1e-5) ** -0.5
    return [value * scale for value in x]


class MicroGPT:
    """A tiny GPT model with parameters represented by Value objects."""

    def __init__(self, vocab_size, config=None, weight_data=None):
        self.vocab_size = vocab_size
        self.config = config or ModelConfig()
        # 16 embedding features / 4 heads = 4 features handled by each head.
        self.head_dim = self.config.n_embd // self.config.n_head

        if weight_data is None:
            # Training starts from small random parameters so different neurons
            # can learn different roles instead of remaining identical.
            self.state_dict = self._initialize_weights()
        else:
            # Loading a snapshot wraps plain JSON floats in Value again. Inference
            # uses only .data, but this also allows resumed training if desired.
            self.state_dict = {
                name: [[Value(value) for value in row] for row in rows]
                for name, rows in weight_data.items()
            }
        # Flatten every matrix into one list so the optimizer can update all 4,800
        # scalar parameters uniformly without knowing which matrix owns them.
        self.parameters = [
            value
            for matrix in self.state_dict.values()
            for row in matrix
            for value in row
        ]

    def _matrix(self, output_size, input_size, std=0.08):
        # A small standard deviation keeps initial activations from exploding.
        # For _matrix(2, 3), the shape is two rows by three columns.
        return [
            [Value(random.gauss(0, std)) for _ in range(input_size)]
            for _ in range(output_size)
        ]

    def _initialize_weights(self):
        config = self.config
        # Rows select outputs and columns consume inputs. At the defaults and a
        # 46-token vocabulary, wte is [46, 16], while lm_head maps [16] back to
        # 46 next-token scores. Training learns every entry in these matrices.
        weights = {
            # Token embedding table: row 10, for example, is the learned vector
            # for vocabulary token 10. Only the selected row is used per call.
            'wte': self._matrix(self.vocab_size, config.n_embd),
            # Position embedding table: row 0 means "first context position",
            # row 1 means "second", etc. This supplies ordering information.
            'wpe': self._matrix(config.block_size, config.n_embd),
            # Language-model head: converts the final representation into one
            # score per possible next character (including the end token).
            'lm_head': self._matrix(self.vocab_size, config.n_embd),
        }
        for layer in range(config.n_layer):
            # Wq, Wk, and Wv are separate learned views of the same input x:
            #
            #   query = Wq*x  -> what information does the current token seek?
            #   key   = Wk*x  -> what information does this token advertise?
            #   value = Wv*x  -> what information should it contribute if chosen?
            #
            # At the defaults all three matrices are [16, 16], so each produces
            # a 16-number vector. They start random and learn their meanings solely
            # from gradients caused by wrong next-character predictions.
            weights[f'layer{layer}.attn_wq'] = self._matrix(config.n_embd, config.n_embd)
            weights[f'layer{layer}.attn_wk'] = self._matrix(config.n_embd, config.n_embd)
            weights[f'layer{layer}.attn_wv'] = self._matrix(config.n_embd, config.n_embd)
            # After the heads are concatenated, Wo lets information from all heads
            # interact and projects the combined [16] vector back into model space.
            weights[f'layer{layer}.attn_wo'] = self._matrix(config.n_embd, config.n_embd)
            # The feed-forward network expands each position from 16 to 64 features,
            # applies ReLU, then compresses it back to 16.
            weights[f'layer{layer}.mlp_fc1'] = self._matrix(4 * config.n_embd, config.n_embd)
            weights[f'layer{layer}.mlp_fc2'] = self._matrix(config.n_embd, 4 * config.n_embd)
        return weights

    def new_cache(self):
        # Each layer gets a growing list of past keys and values. After reading
        # 3 tokens in the default one-layer model, keys[0] and values[0] each
        # contain 3 vectors of width 16. Inference can reuse them as it generates.
        return (
            [[] for _ in range(self.config.n_layer)],
            [[] for _ in range(self.config.n_layer)],
        )

    def forward(self, token_id, position_id, keys, values):
        config = self.config
        # A token embedding describes *what* character this is, while a position
        # embedding describes *where* it occurs. For token_id=10 at position 0,
        # two learned [16] vectors are added to create this token's representation.
        token_embedding = self.state_dict['wte'][token_id]
        position_embedding = self.state_dict['wpe'][position_id]
        # Element-wise addition keeps the width at 16. A toy two-wide example is
        # token=[0.2, -0.1] + position=[0.5, 0.3] -> x=[0.7, 0.2].
        x = [token + position for token, position in zip(token_embedding, position_embedding)]
        x = rmsnorm(x)

        for layer in range(config.n_layer):
            # Keep the input so attention can learn a useful change instead of
            # reconstructing everything; this is the Transformer's residual path.
            residual = x
            x = rmsnorm(x)
            # Q, K, and V are not stored facts or literal characters. They are
            # learned numeric projections of x. Their roles come from how they are
            # used below: Q and K decide relevance; V supplies the resulting data.
            query = linear(x, self.state_dict[f'layer{layer}.attn_wq'])
            key = linear(x, self.state_dict[f'layer{layer}.attn_wk'])
            value = linear(x, self.state_dict[f'layer{layer}.attn_wv'])
            # Cache this position's K and V. Its Q is not cached because a query is
            # needed only once: when that position asks which prefix tokens matter.
            # Keys and values, in contrast, may be consulted by every later token.
            keys[layer].append(key)
            values[layer].append(value)

            attention = []
            for head in range(config.n_head):
                # Four default heads split a 16-wide vector into four independent
                # 4-wide slices. A head can specialize in a different relationship.
                start = head * self.head_dim
                query_head = query[start:start + self.head_dim]
                # At position 2, key_head/value_head contain three 4-number vectors:
                # one each for positions 0, 1, and the current position 2.
                key_head = [item[start:start + self.head_dim] for item in keys[layer]]
                value_head = [item[start:start + self.head_dim] for item in values[layer]]
                attention_logits = [
                    # At position 2 there are 3 cached keys, producing 3 scores:
                    # one for positions 0, 1, and 2. Future positions are absent
                    # from the cache, which makes this attention causal.
                    sum(query_head[j] * key_head[t][j] for j in range(self.head_dim))
                    # Dividing by sqrt(head_dim) prevents larger vector widths from
                    # making dot products so extreme that softmax becomes saturated.
                    / self.head_dim**0.5
                    for t in range(len(key_head))
                ]
                # A larger Q dot K score means the current position's query matches
                # that earlier position's key more closely. For raw scores
                # [0.2, 2.1, 0.5], softmax might assign roughly [0.11, 0.73, 0.16].
                attention_weights = softmax(attention_logits)
                # Example weights [0.1, 0.7, 0.2] mix 10%, 70%, and 20% of the
                # three value vectors into this head's output.
                head_output = [
                    # Apply the same per-position weight to every feature of its V.
                    # If weights=[0.25, 0.75] and one V feature is [2, 6], that
                    # output feature is 0.25*2 + 0.75*6 = 5.
                    sum(
                        attention_weights[t] * value_head[t][j]
                        for t in range(len(value_head))
                    )
                    for j in range(self.head_dim)
                ]
                # Concatenation restores model width: four head outputs of width 4
                # become one attention vector of width 16.
                attention.extend(head_output)

            x = linear(attention, self.state_dict[f'layer{layer}.attn_wo'])
            # The skip connection provides a direct path for both information and
            # gradients: output = learned_attention_update + original_input.
            x = [value + residual_value for value, residual_value in zip(x, residual)]

            # The MLP processes the attended information independently at this
            # position: [16] -> [64] -> ReLU -> [16], followed by another skip.
            residual = x
            x = rmsnorm(x)
            x = linear(x, self.state_dict[f'layer{layer}.mlp_fc1'])
            # Negative expanded features become zero; positive features pass on.
            x = [value.relu() for value in x]
            x = linear(x, self.state_dict[f'layer{layer}.mlp_fc2'])
            x = [value + residual_value for value, residual_value in zip(x, residual)]

        # One unnormalized score (logit) per possible next token. With 45 text
        # characters plus BOS/end, the result is a list of 46 scalar Values.
        return linear(x, self.state_dict['lm_head'])

    def export_weights(self):
        # Autograd history belongs only to the current forward/backward pass. A
        # snapshot needs just the learned numeric data, preserving matrix nesting.
        return {
            name: [[value.data for value in row] for row in rows]
            for name, rows in self.state_dict.items()
        }
