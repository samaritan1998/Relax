# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy
from argparse import Namespace
from typing import Any

import numpy as np
import ray
import torch

from relax.engine.rollout.base_types import RolloutFnTrainOutput
from relax.utils.async_utils import run
from relax.utils.data.data import Dataset
from relax.utils.data.mask_utils import MultiTurnLossMaskGenerator
from relax.utils.data.processing_utils import load_processor, load_tokenizer
from relax.utils.logging_utils import get_logger
from relax.utils.multimodal.config import MultimodalConfig
from relax.utils.types import Sample
from relax.utils.utils import convert_samples_to_train_data, transfer_batch_to_data_system


__all__ = ["generate_rollout", "get_sft_debug_data"]

logger = get_logger(__name__)

TOKENIZER = None
PROCESSOR = None
DATASET = None
MASK_GENERATOR = None
SAMPLE_PRINTED = False


def _get_tokenizer(args: Namespace):
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    return TOKENIZER


def _get_processor(args: Namespace):
    global PROCESSOR
    if PROCESSOR is None:
        PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)
    return PROCESSOR


def _get_mask_generator(args: Namespace) -> MultiTurnLossMaskGenerator:
    global MASK_GENERATOR
    if MASK_GENERATOR is None:
        MASK_GENERATOR = MultiTurnLossMaskGenerator(
            _get_tokenizer(args),
            tokenizer_type=args.loss_mask_type,
        )
    return MASK_GENERATOR


def _get_dataset(args: Namespace) -> Dataset:
    global DATASET
    if DATASET is None:
        DATASET = Dataset(
            args.prompt_data,
            tokenizer=_get_tokenizer(args),
            processor=_get_processor(args),
            max_length=args.rollout_max_prompt_len,
            prompt_key=args.input_key,
            multimodal_keys=args.multimodal_keys,
            label_key=args.label_key,
            metadata_key=args.metadata_key,
            system_prompt=args.system_prompt,
            tool_key=args.tool_key,
            apply_chat_template=False,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
            use_audio_in_video=args.use_audio_in_video,
            seed=args.rollout_seed,
            multimodal_config=MultimodalConfig.from_args(args),
        )
        if len(DATASET) == 0:
            raise ValueError(f"SFT prompt_data has no usable samples: {args.prompt_data}")
    return DATASET


def _to_list_ids(input_ids: Any) -> list[int]:
    if isinstance(input_ids, torch.Tensor):
        return input_ids.tolist()
    if isinstance(input_ids, np.ndarray):
        return input_ids.tolist()
    return list(input_ids)


def _to_tensor_or_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return value


def _messages_from_sample(sample: Sample) -> list[dict[str, Any]]:
    prompt = copy.deepcopy(sample.prompt)

    if isinstance(prompt, str):
        if sample.label is None:
            raise ValueError("SFT sample has a string prompt but no label to form the assistant response.")
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": sample.label},
        ]
    elif isinstance(prompt, list):
        messages = prompt
        has_assistant = any(message.get("role") == "assistant" for message in messages)
        if not has_assistant:
            if sample.label is None:
                raise ValueError("SFT sample has no assistant turn and no label.")
            messages = messages + [{"role": "assistant", "content": sample.label}]
    else:
        raise TypeError(f"Unsupported SFT prompt type: {type(prompt)}")

    return messages


def _render_messages(args: Namespace, messages: list[dict[str, Any]], tools: list[dict] | None) -> str:
    tokenizer = _get_tokenizer(args)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        tools=tools,
        return_dict=False,
    )


def _build_sft_sample(args: Namespace, sample: Sample) -> Sample:
    messages = _messages_from_sample(sample)
    tools = sample.metadata.get("tools", None) if sample.metadata else None
    mask_generator = _get_mask_generator(args)

    if args.multimodal_keys is not None:
        processor = _get_processor(args)
        if processor is None:
            raise RuntimeError("--multimodal-keys is set, but no HuggingFace processor was loaded.")

        rendered_text = _render_messages(args, messages, tools)
        processor_output = processor(
            text=rendered_text,
            use_audio_in_video=args.use_audio_in_video,
            return_mm_token_type_ids=False,
            **(sample.multimodal_inputs or {}),
        )
        input_ids = _to_list_ids(processor_output["input_ids"][0])
        token_ids, loss_mask = mask_generator.get_loss_mask_with_multimodal_alignment(
            messages,
            input_ids,
            tools=tools,
        )
        sample.multimodal_train_inputs = {
            key: _to_tensor_or_value(value)
            for key, value in processor_output.items()
            if key not in ("input_ids", "attention_mask")
        } or None
    else:
        token_ids, loss_mask = mask_generator.get_loss_mask(messages, tools=tools)
        sample.multimodal_train_inputs = None

    if len(token_ids) != len(loss_mask):
        raise ValueError(
            f"SFT sample produced mismatched token_ids/loss_mask: {len(token_ids)=}, {len(loss_mask)=}"
        )

    max_context_len = getattr(args, "rollout_max_context_len", None)
    if max_context_len is not None and len(token_ids) > max_context_len:
        raise ValueError(
            f"SFT sample length {len(token_ids)} exceeds --rollout-max-context-len {max_context_len}."
        )

    response_length = mask_generator.get_response_lengths([loss_mask])[0]
    if response_length <= 0:
        raise ValueError(f"SFT sample has no supervised assistant tokens: {messages=}")

    sample.tokens = token_ids
    sample.rollout_tokens = token_ids
    sample.response_length = response_length
    sample.reward = 0.0
    sample.loss_mask = loss_mask[-response_length:]
    sample.status = Sample.Status.COMPLETED
    return sample


def _get_global_samples(args: Namespace, rollout_id: int, num_samples: int) -> list[Sample]:
    dataset = _get_dataset(args)
    samples: list[Sample] = []
    cursor = rollout_id * num_samples

    while len(samples) < num_samples:
        epoch_id = cursor // len(dataset)
        offset = cursor % len(dataset)
        if args.rollout_shuffle:
            dataset.shuffle(epoch_id)

        take = min(num_samples - len(samples), len(dataset) - offset)
        samples.extend(copy.deepcopy(dataset.samples[offset : offset + take]))
        cursor += take

    return samples


def _build_sft_samples(args: Namespace, rollout_id: int, num_samples: int) -> list[Sample]:
    global SAMPLE_PRINTED

    samples = [
        _build_sft_sample(args, sample)
        for sample in _get_global_samples(args, rollout_id, num_samples)
    ]
    for i, sample in enumerate(samples):
        sample.group_index = rollout_id * num_samples + i
        sample.index = rollout_id * num_samples + i

    if samples and not SAMPLE_PRINTED:
        first = samples[0]
        mm_keys = sorted(first.multimodal_train_inputs.keys()) if first.multimodal_train_inputs else []
        logger.info(
            "sft_rollout example: "
            f"tokens={len(first.tokens)}, response_length={first.response_length}, "
            f"loss_tokens={sum(first.loss_mask or [])}, multimodal_train_input_keys={mm_keys}"
        )
        SAMPLE_PRINTED = True

    return samples


def get_sft_debug_data(
    args: Namespace,
    rollout_id: int,
    batch_size: int,
    dp_rank: int,
    dp_size: int,
) -> dict[str, Any]:
    global_batch_size = batch_size * dp_size
    samples = _build_sft_samples(args, rollout_id, global_batch_size)
    local_samples = samples[dp_rank * batch_size : (dp_rank + 1) * batch_size]
    return convert_samples_to_train_data(args, local_samples)


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    data_system_client: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput:
    if evaluation:
        raise NotImplementedError("SFT rollout does not support evaluation.")

    raw_sample_groups = ray.get(data_source.get_samples.remote(args.rollout_batch_size))
    samples = [_build_sft_sample(args, group[0]) for group in raw_sample_groups]
    for i, sample in enumerate(samples):
        sample.group_index = rollout_id * args.rollout_batch_size + i
        sample.index = rollout_id * args.rollout_batch_size + i

    run(
        transfer_batch_to_data_system(
            args,
            [[sample] for sample in samples],
            len(samples),
            rollout_id,
            data_system_client,
        )
    )
    return RolloutFnTrainOutput(samples=[[sample] for sample in samples], metrics={})
