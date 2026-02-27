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

"""Evaluates a trained Transformer on UTM-generated sequences."""

import csv
import functools
import math
from typing import Any

from absl import app
from absl import flags
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from data import chomsky_data_generator as chomsky_dg_lib
from data import ctw_data_generator as ctw_dg_lib
from data import utm_data_generator as utm_dg_lib
from data import utms as utms_lib
from models import ctw as ctw_lib
from models import transformer

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "params_path",
    "params.npz",
    "Path to model parameters saved by train.py.",
)
flags.DEFINE_bool(
    "compute_ctw_regret",
    False,
    (
        "If True, evaluate a CTW predictor on the same sequences and report "
        "regret (NLL_model - NLL_CTW)."
    ),
)
flags.DEFINE_integer(
    "ctw_depth",
    24,
    "Context depth for CTW predictor.",
)
flags.DEFINE_integer(
    "ctw_max_capacity",
    2**27,
    "Maximum number of contexts for CTW predictor table.",
)
flags.DEFINE_enum(
    "ctw_first_tokens_method",
    "pad_with_zeros",
    ["pad_with_zeros", "build_tree"],
    "How CTW handles first tokens before full depth is available.",
)
flags.DEFINE_list(
    "eval_seq_lengths",
    ["256", "1024"],
    "Comma-separated sequence lengths to evaluate on.",
)
flags.DEFINE_integer(
    "num_eval_sequences",
    6000,
    "Total number of sequences to evaluate per sequence length.",
)
flags.DEFINE_integer("batch_size", 128, "Batch size used for evaluation.")
flags.DEFINE_integer("memory_size", 10, "UTM memory size.")
flags.DEFINE_integer(
    "maximum_steps",
    100,
    "Maximum UTM execution steps per sampled program.",
)
flags.DEFINE_integer(
    "maximum_program_length",
    100,
    "Maximum sampled program length.",
)
flags.DEFINE_enum(
    "position_encoding_type",
    "sinusoidal",
    ["sinusoidal", "relative_bias"],
    "Transformer position encoding strategy.",
)
flags.DEFINE_integer("embedding_dim", 64, "Transformer embedding dimension.")
flags.DEFINE_integer("num_layers", 4, "Number of decoder layers.")
flags.DEFINE_integer("num_heads", 8, "Number of attention heads.")
flags.DEFINE_integer(
    "widening_factor", 4, "FFN widening factor relative to embedding_dim."
)
flags.DEFINE_integer(
    "relative_attention_num_buckets",
    32,
    "Number of buckets for relative attention bias.",
)
flags.DEFINE_integer(
    "relative_attention_max_distance",
    128,
    "Max distance used by relative attention bucketization.",
)
flags.DEFINE_integer(
    "train_seq_length",
    256,
    "Reference train length used for head/tail generalization split.",
)
flags.DEFINE_integer("eval_seed", 1, "Seed used for evaluation data generation.")
flags.DEFINE_enum(
    "data_source",
    "utm",
    ["utm", "ctw", "chomsky"],
    "Data generator to use for evaluation.",
)
# Chomsky-specific flags (used when data_source=chomsky).
flags.DEFINE_enum(
    "chomsky_task",
    "parity_check",
    chomsky_dg_lib.ORDERED_TASKS,
    "Chomsky hierarchy task to evaluate on.",
)
flags.DEFINE_integer(
    "chomsky_max_input_length", 10, "Maximum input length for Chomsky tasks."
)
flags.DEFINE_boolean(
    "chomsky_use_delimiters",
    True,
    "Whether to include input/output delimiters in Chomsky sequences.",
)
# CTW/VOMS-specific flags (used when data_source=ctw).
flags.DEFINE_integer(
    "ctw_max_depth", 5, "Maximum tree depth for the CTW/VOMS data generator."
)
flags.DEFINE_string(
    "position_metrics_csv",
    "",
    (
        "Optional path to write per-position metrics CSV. "
        "If empty, no CSV is written."
    ),
)


def _parse_eval_lengths(lengths: list[str]) -> list[int]:
    parsed = []
    for value in lengths:
        seq_len = int(value)
        if seq_len <= 0:
            raise ValueError(f"All eval lengths must be > 0, got {seq_len}")
        parsed.append(seq_len)
    return parsed


def _infer_vocab_size(params: hk.Params) -> int:
    """Infers vocab_size from the embedding matrix shape in a checkpoint."""
    def _search(obj: Any) -> int | None:
        if isinstance(obj, dict):
            if "embeddings" in obj:
                return obj["embeddings"].shape[0]
            for v in obj.values():
                result = _search(v)
                if result is not None:
                    return result
        return None

    result = _search(params)
    if result is None:
        raise ValueError(
            "Could not infer vocab_size from checkpoint params. "
            "Check that the params file was saved by train.py."
        )
    return result


def _build_model(config: transformer.TransformerConfig) -> hk.Transformed:
    return hk.transform(
        functools.partial(transformer.transformer_decoder, config=config)
    )


def _load_params(path: str) -> hk.Params:
    with np.load(path, allow_pickle=True) as data:
        params = {}
        for key in data.files:
            value = data[key]
            # train.py uses np.savez(**params) where each top-level leaf is a nested
            # dict serialized as a 0-d object array.
            if value.dtype == object and value.shape == ():
                params[key] = value.item()
            else:
                params[key] = value
    return params


def _make_token_log_probs_fn(model: hk.Transformed) -> Any:
    """Returns a jitted function for per-token true log probabilities."""

    @jax.jit
    def token_log_probs(
        params: hk.Params,
        sequences: jax.Array,
    ) -> jax.Array:
        conditionals = model.apply(params=params, targets=sequences, rng=None)
        true_log_probs = jnp.take_along_axis(
            conditionals, sequences[..., None], axis=-1
        )[..., 0]
        return true_log_probs

    return token_log_probs


def _build_generator(seq_length: int) -> Any:
    """Builds a data generator for the configured data_source."""
    rng = np.random.default_rng(seed=FLAGS.eval_seed)
    if FLAGS.data_source == "utm":
        program_sampler = utms_lib.FastSampler(rng=rng)
        utm = utms_lib.BrainPhoqueUTM(program_sampler)
        return utm_dg_lib.UTMDataGenerator(
            batch_size=FLAGS.batch_size,
            seq_length=seq_length,
            rng=rng,
            utm=utm,
            memory_size=FLAGS.memory_size,
            maximum_steps=FLAGS.maximum_steps,
            tokenizer=utm_dg_lib.Tokenizer.ASCII,
            maximum_program_length=FLAGS.maximum_program_length,
        )
    elif FLAGS.data_source == "ctw":
        return ctw_dg_lib.CTWGenerator(
            batch_size=FLAGS.batch_size,
            seq_length=seq_length,
            rng=rng,
            max_depth=FLAGS.ctw_max_depth,
        )
    elif FLAGS.data_source == "chomsky":
        return chomsky_dg_lib.ChomskyDataGenerator(
            task_str=FLAGS.chomsky_task,
            max_input_length=FLAGS.chomsky_max_input_length,
            use_delimiters=FLAGS.chomsky_use_delimiters,
            batch_size=FLAGS.batch_size,
            seq_length=seq_length,
            rng=rng,
        )


def _evaluate_once(
    params: hk.Params,
    token_log_probs_fn: Any,
    seq_length: int,
) -> dict[str, Any]:
    generator = _build_generator(seq_length)

    total_sequences = FLAGS.num_eval_sequences
    num_batches = math.ceil(total_sequences / FLAGS.batch_size)
    keep_last = total_sequences - (num_batches - 1) * FLAGS.batch_size

    position_nll_sum = np.zeros(seq_length, dtype=np.float64)
    position_count = np.zeros(seq_length, dtype=np.int64)
    total_nll = 0.0
    total_tokens = 0

    split = min(max(FLAGS.train_seq_length, 0), seq_length)
    head_nll = 0.0
    head_count = 0
    tail_nll = 0.0
    tail_count = 0
    ctw_total_nll = 0.0
    ctw_position_nll_sum = np.zeros(seq_length, dtype=np.float64)
    ctw_head_nll = 0.0
    ctw_tail_nll = 0.0

    ctw_predictor = None
    if FLAGS.compute_ctw_regret:
        first_tokens_method = (
            ctw_lib.FirstTokensMethod.PAD_WITH_ZEROS
            if FLAGS.ctw_first_tokens_method == "pad_with_zeros"
            else ctw_lib.FirstTokensMethod.BUILD_TREE
        )
        ctw_predictor = ctw_lib.CTWPredictor(
            depth=FLAGS.ctw_depth,
            batched_update=True,
            max_capacity=FLAGS.ctw_max_capacity,
            first_tokens_method=first_tokens_method,
        )

    for batch_index in range(num_batches):
        batch_one_hot, logs = generator.sample()
        batch_one_hot = np.asarray(batch_one_hot, dtype=np.uint8)
        batch = np.argmax(batch_one_hot, axis=-1).astype(np.int32)

        if batch_index == num_batches - 1 and keep_last < FLAGS.batch_size:
            batch = batch[:keep_last]
            batch_one_hot = batch_one_hot[:keep_last]
            if "loss_mask" in logs:
                loss_mask = np.asarray(logs["loss_mask"][:keep_last], dtype=bool)
            else:
                loss_mask = np.zeros(batch.shape, dtype=bool)
        else:
            if "loss_mask" in logs:
                loss_mask = np.asarray(logs["loss_mask"], dtype=bool)
            else:
                loss_mask = np.zeros(batch.shape, dtype=bool)

        true_log_probs = token_log_probs_fn(params=params, sequences=batch)
        nll = -np.asarray(true_log_probs, dtype=np.float64)
        valid = ~loss_mask

        nll_masked = nll * valid
        total_nll += float(np.sum(nll_masked))
        total_tokens += int(np.sum(valid))

        position_nll_sum += np.sum(nll_masked, axis=0)
        position_count += np.sum(valid, axis=0)

        if split > 0:
            head_valid = valid[:, :split]
            head_nll += float(np.sum(nll[:, :split] * head_valid))
            head_count += int(np.sum(head_valid))
        if split < seq_length:
            tail_valid = valid[:, split:]
            tail_nll += float(np.sum(nll[:, split:] * tail_valid))
            tail_count += int(np.sum(tail_valid))

        if ctw_predictor is not None:
            ctw_conditionals, _, _, _, _ = ctw_predictor.update(
                params={},
                sequences=batch_one_hot,
                contexts=None,
                return_only_marginals=False,
            )
            ctw_true_log_probs = np.take_along_axis(
                ctw_conditionals, batch[..., None], axis=-1
            )[..., 0]
            ctw_nll = -np.asarray(ctw_true_log_probs, dtype=np.float64)
            ctw_nll_masked = ctw_nll * valid
            ctw_total_nll += float(np.sum(ctw_nll_masked))
            ctw_position_nll_sum += np.sum(ctw_nll_masked, axis=0)
            if split > 0:
                ctw_head_nll += float(np.sum(ctw_nll[:, :split] * head_valid))
            if split < seq_length:
                ctw_tail_nll += float(np.sum(ctw_nll[:, split:] * tail_valid))

    avg_nll = total_nll / total_tokens
    bits_per_token = avg_nll / np.log(2.0)
    perplexity = float(np.exp(avg_nll))
    position_avg_nll = np.divide(
        position_nll_sum,
        np.maximum(position_count, 1),
        dtype=np.float64,
    )

    result = {
        "seq_length": seq_length,
        "num_sequences": total_sequences,
        "num_tokens": total_tokens,
        "avg_nll_nats": float(avg_nll),
        "bits_per_token": float(bits_per_token),
        "perplexity": perplexity,
        "position_avg_nll": position_avg_nll,
        "position_count": position_count,
    }
    if head_count > 0:
        result["head_avg_nll_nats"] = head_nll / head_count
        result["head_tokens"] = head_count
    if tail_count > 0:
        result["tail_avg_nll_nats"] = tail_nll / tail_count
        result["tail_tokens"] = tail_count
    if ctw_predictor is not None:
        ctw_avg_nll = ctw_total_nll / total_tokens
        ctw_position_avg_nll = np.divide(
            ctw_position_nll_sum,
            np.maximum(position_count, 1),
            dtype=np.float64,
        )
        result["ctw_avg_nll_nats"] = float(ctw_avg_nll)
        result["avg_regret_nats_per_token"] = float(avg_nll - ctw_avg_nll)
        result["cumulative_regret_nats"] = float(total_nll - ctw_total_nll)
        result["position_avg_ctw_nll"] = ctw_position_avg_nll
        result["position_avg_regret_nats"] = position_avg_nll - ctw_position_avg_nll
        if head_count > 0:
            result["head_ctw_avg_nll_nats"] = ctw_head_nll / head_count
            result["head_avg_regret_nats"] = (
                result["head_avg_nll_nats"] - result["head_ctw_avg_nll_nats"]
            )
        if tail_count > 0:
            result["tail_ctw_avg_nll_nats"] = ctw_tail_nll / tail_count
            result["tail_avg_regret_nats"] = (
                result["tail_avg_nll_nats"] - result["tail_ctw_avg_nll_nats"]
            )
    return result


def _write_position_(
    path: str,
    results: list[dict[str, Any]],
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seq_length",
                "position",
                "avg_nll_nats",
                "ctw_avg_nll_nats",
                "avg_regret_nats",
                "count",
            ]
        )
        for run in results:
            seq_length = run["seq_length"]
            avg = run["position_avg_nll"]
            ctw_avg = run.get("position_avg_ctw_nll")
            regret = run.get("position_avg_regret_nats")
            count = run["position_count"]
            for i in range(seq_length):
                row = [seq_length, i + 1, float(avg[i])]
                if ctw_avg is not None and regret is not None:
                    row.extend([float(ctw_avg[i]), float(regret[i])])
                else:
                    row.extend(["", ""])
                row.append(int(count[i]))
                writer.writerow(row)


def main(_) -> None:
    eval_lengths = _parse_eval_lengths(FLAGS.eval_seq_lengths)
    params = _load_params(FLAGS.params_path)

    config = transformer.TransformerConfig(
        vocab_size=_infer_vocab_size(params),
        embedding_dim=FLAGS.embedding_dim,
        num_layers=FLAGS.num_layers,
        num_heads=FLAGS.num_heads,
        widening_factor=FLAGS.widening_factor,
        position_encoding_type=FLAGS.position_encoding_type,
        relative_attention_num_buckets=FLAGS.relative_attention_num_buckets,
        relative_attention_max_distance=FLAGS.relative_attention_max_distance,
    )
    model = _build_model(config)
    token_log_probs_fn = _make_token_log_probs_fn(model)

    runs = []
    logging.info("Evaluating checkpoint %s", FLAGS.params_path)
    for seq_length in eval_lengths:
        run = _evaluate_once(
            params=params,
            token_log_probs_fn=token_log_probs_fn,
            seq_length=seq_length,
        )
        runs.append(run)
        logging.info(
            (
                "[L=%d] nll=%.6f nats/token, bpt=%.6f, ppl=%.6f, "
                "tokens=%d, sequences=%d"
            ),
            run["seq_length"],
            run["avg_nll_nats"],
            run["bits_per_token"],
            run["perplexity"],
            run["num_tokens"],
            run["num_sequences"],
        )
        if "ctw_avg_nll_nats" in run:
            logging.info(
                (
                    "[L=%d] ctw_nll=%.6f nats/token, "
                    "avg_regret=%.6f nats/token, cumulative_regret=%.6f nats"
                ),
                run["seq_length"],
                run["ctw_avg_nll_nats"],
                run["avg_regret_nats_per_token"],
                run["cumulative_regret_nats"],
            )
        if "head_avg_nll_nats" in run:
            logging.info(
                "[L=%d] head(<=%d): %.6f nats/token over %d tokens",
                run["seq_length"],
                min(FLAGS.train_seq_length, run["seq_length"]),
                run["head_avg_nll_nats"],
                run["head_tokens"],
            )
            if "head_ctw_avg_nll_nats" in run:
                logging.info(
                    (
                        "[L=%d] head(<=%d): ctw=%.6f nats/token, "
                        "regret=%.6f nats/token"
                    ),
                    run["seq_length"],
                    min(FLAGS.train_seq_length, run["seq_length"]),
                    run["head_ctw_avg_nll_nats"],
                    run["head_avg_regret_nats"],
                )
        if "tail_avg_nll_nats" in run:
            logging.info(
                "[L=%d] tail(>%d): %.6f nats/token over %d tokens",
                run["seq_length"],
                min(FLAGS.train_seq_length, run["seq_length"]),
                run["tail_avg_nll_nats"],
                run["tail_tokens"],
            )
            if "tail_ctw_avg_nll_nats" in run:
                logging.info(
                    (
                        "[L=%d] tail(>%d): ctw=%.6f nats/token, "
                        "regret=%.6f nats/token"
                    ),
                    run["seq_length"],
                    min(FLAGS.train_seq_length, run["seq_length"]),
                    run["tail_ctw_avg_nll_nats"],
                    run["tail_avg_regret_nats"],
                )

    if FLAGS.position_metrics_csv:
        _write_position_(FLAGS.position_metrics_csv, runs)
        logging.info("Wrote per-position metrics to %s", FLAGS.position_metrics_csv)


if __name__ == "__main__":
    app.run(main)
