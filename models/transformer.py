# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Transformer model."""

import dataclasses

import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(kw_only=True)
class TransformerConfig:
  """Hyperparameters used in the Transformer architectures."""

  # Vocabulary size.
  vocab_size: int
  # The dimension of the first embedding.
  embedding_dim: int = 64
  # The number of multi-head attention layers.
  num_layers: int = 4
  # The number of heads per layer.
  num_heads: int = 8
  # The parameter initialization scale for the embeddings.
  emb_init_scale: float = 0.02
  # How much larger the hidden layer of the feedforward network should be
  # compared to the `embedding_dim`.
  widening_factor: int = 4
  # Positional mechanism used by the decoder.
  # - "sinusoidal": absolute sinusoidal encodings (paper baseline).
  # - "relative_bias": causal relative position bias on attention logits.
  position_encoding_type: str = 'sinusoidal'
  # Number of buckets for T5-style relative position bias.
  relative_attention_num_buckets: int = 32
  # Maximum distance for non-linear bucket scaling in relative bias.
  relative_attention_max_distance: int = 128


def _relative_position_bucket(
    relative_position: jax.Array,
    num_buckets: int,
    max_distance: int,
) -> jax.Array:
  """Maps relative positions to logarithmic buckets (causal variant)."""
  num_buckets_exact = num_buckets // 2
  max_exact = num_buckets_exact
  max_distance = max(max_distance, max_exact + 1)

  # For causal attention, valid relative positions are <= 0 (k <= q).
  distance = -relative_position
  distance = jnp.maximum(distance, 0)

  is_small = distance < max_exact
  safe_distance = jnp.maximum(distance, 1)
  log_scale = jnp.log(safe_distance / max_exact) / jnp.log(
      max_distance / max_exact
  )
  large_bucket = max_exact + (
      log_scale * (num_buckets - max_exact)
  ).astype(jnp.int32)
  large_bucket = jnp.minimum(large_bucket, num_buckets - 1)
  return jnp.where(is_small, distance, large_bucket)


class MultiHeadDotProductAttention(hk.Module):
  """Multi-head dot-product attention (Vaswani et al., 2017)."""

  def __init__(
      self,
      num_heads: int,
      num_hiddens_per_head: int,
      use_relative_position_bias: bool = False,
      relative_attention_num_buckets: int = 32,
      relative_attention_max_distance: int = 128,
      name: str | None = None,
  ) -> None:
    """Initializes the attention module.

    Args:
      num_heads: Number of heads to use.
      num_hiddens_per_head: Number of hidden neurons per head.
      use_relative_position_bias: Whether to add relative attention bias terms.
      relative_attention_num_buckets: Number of relative position buckets.
      relative_attention_max_distance: Max distance used by bucket mapping.
      name: Name of the module.
    """
    super().__init__(name=name)
    self._num_heads = num_heads
    self._num_hiddens_per_head = num_hiddens_per_head
    self._use_relative_position_bias = use_relative_position_bias
    self._relative_attention_num_buckets = relative_attention_num_buckets
    self._relative_attention_max_distance = relative_attention_max_distance

  def _relative_position_bias(
      self,
      query_length: int,
      key_length: int,
  ) -> jax.Array:
    """Returns per-head relative attention bias, shape [1, H, Q, K]."""
    query_positions = jnp.arange(query_length, dtype=jnp.int32)[:, None]
    key_positions = jnp.arange(key_length, dtype=jnp.int32)[None, :]
    relative_position = key_positions - query_positions
    bucket = _relative_position_bucket(
        relative_position=relative_position,
        num_buckets=self._relative_attention_num_buckets,
        max_distance=self._relative_attention_max_distance,
    )
    relative_attention_bias = hk.get_parameter(
        'relative_attention_bias',
        shape=(self._num_heads, self._relative_attention_num_buckets),
        init=hk.initializers.TruncatedNormal(stddev=0.02),
    )
    values = jnp.take(relative_attention_bias, bucket, axis=1)
    return values[None, ...]

  def __call__(
      self,
      inputs_q: jax.Array,
      inputs_kv: jax.Array,
      mask: jax.Array | None = None,
  ) -> jax.Array:
    """Returns the output of the multi-head attention."""
    batch_size, query_length, embedding_size = inputs_q.shape
    _, key_length, _ = inputs_kv.shape

    num_hiddens = self._num_hiddens_per_head * self._num_heads
    q = hk.Linear(num_hiddens, with_bias=False)(inputs_q)
    k = hk.Linear(num_hiddens, with_bias=False)(inputs_kv)
    v = hk.Linear(num_hiddens, with_bias=False)(inputs_kv)
    # The second (sequence) dimension is undefined since it can differ between
    # queries and keys/values when decoding. Also checking that the inputs have
    # the same batch size as the reshape below does not guarantee a failure if
    # they are different.
    q = jnp.reshape(
        q, (batch_size, query_length, self._num_heads, self._num_hiddens_per_head)
    )
    k = jnp.reshape(
        k, (batch_size, key_length, self._num_heads, self._num_hiddens_per_head)
    )
    v = jnp.reshape(
        v, (batch_size, key_length, self._num_heads, self._num_hiddens_per_head)
    )

    # Let b=batch_size, q=query_length, k=key_length, h=num_heads, and d=dim.
    attention = jnp.einsum('bqhd,bkhd->bhqk', q, k)
    attention *= 1.0 / jnp.sqrt(self._num_hiddens_per_head)
    if self._use_relative_position_bias:
      attention += self._relative_position_bias(query_length, key_length)

    if mask is not None:
      attention = jnp.where(mask, attention, jnp.finfo(jnp.float32).min)

    normalized_attention = jnn.softmax(attention)

    output = jnp.einsum('bhqk,bkhd->bqhd', normalized_attention, v)
    output = jnp.reshape(output, (batch_size, query_length, num_hiddens))
    return hk.Linear(embedding_size, with_bias=False)(output)


def sinusoid_position_encoding(
    sequence_length: int,
    hidden_size: int,
    max_timescale: float = 1e4,
) -> np.ndarray:
  """Creates sinusoidal encodings from the original transformer paper.

  The returned values are, for all i < D/2:
    array[pos, i] = sin(pos / (max_timescale^(2*i / D)))
    array[pos, D/2 + i] = cos(pos / (max_timescale^(2*i / D)))

  Args:
    sequence_length: Sequence length.
    hidden_size: Dimension of the positional encoding vectors, D. Should be
      even.
    max_timescale: Maximum timescale for the frequency.

  Returns:
    An array of shape [L, D] if `add_negative` or `keep_positive_side` is
    `False`, else [2 * L, D].
  """
  freqs = np.arange(0, hidden_size + 1, 2)
  inv_freq = max_timescale ** (-freqs / hidden_size)

  pos_seq = np.arange(start=0, stop=sequence_length)

  sinusoid_inp = np.einsum('i,j->ij', pos_seq, inv_freq)
  embeddings = np.concatenate(
      [np.sin(sinusoid_inp), np.cos(sinusoid_inp)], axis=-1
  )
  return embeddings[:, :hidden_size]


def embed_sequences(
    sequences: jax.Array,
    config: TransformerConfig,
) -> jax.Array:
  """Returns embeddings for sequences of tokens."""
  embs_init = hk.initializers.TruncatedNormal(stddev=config.emb_init_scale)
  embeddings_layer = hk.Embed(
      vocab_size=config.vocab_size,
      embed_dim=config.embedding_dim,
      lookup_style=hk.EmbedLookupStyle.ARRAY_INDEX,
      w_init=embs_init,
  )
  embeddings = embeddings_layer(sequences)
  embeddings *= jnp.sqrt(config.embedding_dim)

  if config.position_encoding_type == 'relative_bias':
    return embeddings
  if config.position_encoding_type != 'sinusoidal':
    raise ValueError(
        f'Unknown position_encoding_type: {config.position_encoding_type}'
    )

  _, sequence_length, embedding_size = embeddings.shape
  pos_encodings = sinusoid_position_encoding(
      sequence_length=sequence_length,
      hidden_size=embedding_size,
  )
  return embeddings + pos_encodings


def layer_norm(x: jax.Array) -> jax.Array:
  """Helper function for layer norm."""
  return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)


def shift_right(sequences: jax.Array) -> jax.Array:
  """Right-shift the one-hot encoded input by padding on the temporal axis."""
  bos_array = jnp.zeros((sequences.shape[0], 1), dtype=jnp.uint8)
  padded_sequences = jnp.concatenate([bos_array, sequences], axis=1)
  return padded_sequences[:, :-1]


def transformer_decoder(
    targets: jax.Array,
    config: TransformerConfig,
) -> jax.Array:
  """Returns the transformer decoder output, shape [B, T, V].

  Args:
    targets: The integer target values, shape [B, T].
    config: The config to use for the transformer.
  """
  if config.embedding_dim % config.num_heads != 0:
    raise ValueError(
        'embedding_dim must be divisible by num_heads: '
        f'{config.embedding_dim} vs {config.num_heads}'
    )

  # Right shift the targets to get the inputs (the first token is now a 0).
  inputs = shift_right(targets)

  # Embeds the inputs and adds positional encodings.
  embeddings = embed_sequences(inputs, config)

  batch_size, sequence_length = embeddings.shape[:2]

  # The causal mask is shared across heads.
  causal_mask = np.tril(
      np.ones((batch_size, 1, sequence_length, sequence_length), dtype=bool)
  )

  h = embeddings
  for _ in range(config.num_layers):
    self_attention = MultiHeadDotProductAttention(
        num_heads=config.num_heads,
        num_hiddens_per_head=config.embedding_dim // config.num_heads,
        use_relative_position_bias=config.position_encoding_type
        == 'relative_bias',
        relative_attention_num_buckets=config.relative_attention_num_buckets,
        relative_attention_max_distance=config.relative_attention_max_distance,
    )(inputs_q=h, inputs_kv=h, mask=causal_mask)
    attention = layer_norm(h + self_attention)

    # Position-wise feedforward network.
    h = hk.Linear(config.embedding_dim * config.widening_factor)(attention)
    h = jnn.gelu(h)
    h = hk.Linear(config.embedding_dim)(h)
    h = layer_norm(h + attention)

  logits = hk.Linear(config.vocab_size)(h)
  return jnn.log_softmax(logits, axis=-1)
