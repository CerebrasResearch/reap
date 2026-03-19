"""Convert datasets to transformers BatchEncoded or vLLM TokensPrompt formats.

We follow the OpenAI spec for conversational datasets.

ie..,
messages = [
    {"role": "system", "content": "You are AGI"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "What is my purpose?"},
]

Includes the ability to select from specific categories within the dataset and convert
the dataset into either a language modelling dataset with attention applied to every
token or a prompt-completion dataset for training on completions only with SFTTrainer.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import uuid
import json
import re
import random
import logging


import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, BatchEncoding
from vllm import TokensPrompt


logger = logging.getLogger(__name__)


def _maybe_json_load(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _normalize_message_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                else:
                    parts.append(json.dumps(item, sort_keys=True))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


@dataclass
class CompositeDatasetComponent:
    """A single component of a composite dataset specification.

    Attributes:
        name: HuggingFace dataset name (e.g., "open-r1/Mixture-of-Thoughts").
        split: HF dataset split to load (e.g., "train", "tool"). None means use
               the default split from DatasetArgs.
        subset: HF dataset config/subset name (e.g., "code", "math"). None means
                do not pass a subset to ``load_dataset``.
        num_samples: Number of samples to draw from this component.
    """

    name: str
    split: str | None
    subset: str | None
    num_samples: int


# Regex to parse a single component: <name>[<subset>](<split>):<num_samples>
# Examples:
#   "theblackcat102/evol-codealpaca-v1:4096"            -> name="theblackcat102/evol-codealpaca-v1", subset=None, split=None, num_samples=4096
#   "open-r1/Mixture-of-Thoughts[code]:4096"            -> name="open-r1/Mixture-of-Thoughts", subset="code", split=None, num_samples=4096
#   "SWE-bench/SWE-smith-trajectories(tool):4096"      -> name="SWE-bench/SWE-smith-trajectories", subset=None, split="tool", num_samples=4096
#   "dataset[subset](split):4096"                      -> name="dataset", subset="subset", split="split", num_samples=4096
_COMPOSITE_COMPONENT_RE = re.compile(
    r"^(?P<name>[^\[\]()[:,]+)"  # dataset name
    r"(?:\[(?P<subset>[^\]]+)\])?"  # optional [subset]
    r"(?:\((?P<split>[^\)]+)\))?"  # optional (split)
    r":(?P<num_samples>\d+)$"  # :num_samples (required for composite)
)


def parse_composite_dataset_spec(
    spec: str,
    default_split: str = "train",
    default_subset: str | None = None,
) -> list[CompositeDatasetComponent] | None:
    """Parse a composite dataset specification string.

    Returns a list of CompositeDatasetComponent if the spec is composite
    (contains comma-separated entries with :num_samples), or None if the spec
    is a single dataset name (backward-compatible).

    Format: ``name1[subset1](split1):N1,name2:N2,name3[subset3]:N3,...``

    Args:
        spec: The dataset specification string.
        default_split: The default split to use when no split is specified.
        default_subset: The default subset to use when no subset is specified.

    Returns:
        List of parsed components, or None if this is a plain single-dataset name.

    Raises:
        ValueError: If the spec looks like a composite spec but has parse errors.
    """
    # A composite spec must contain at least one colon followed by digits.
    # Single dataset names like "theblackcat102/evol-codealpaca-v1" won't match.
    if ":" not in spec:
        return None

    # Could be a single dataset with a colon in the name (unlikely for HF) —
    # but to be safe, also require at least one comma OR the entire string to
    # match the component pattern.
    parts = [p.strip() for p in spec.split(",")]

    components = []
    for i, part in enumerate(parts):
        m = _COMPOSITE_COMPONENT_RE.match(part)
        if m is None:
            if len(parts) == 1:
                # Single entry that doesn't match composite format — treat as
                # a plain dataset name (backward compatible).
                return None
            raise ValueError(
                f"Failed to parse composite dataset component {i}: '{part}'. "
                f"Expected format: <dataset_name>[<subset>](<split>):<num_samples>. "
                f"Full spec: '{spec}'"
            )
        name = m.group("name").strip()
        subset = m.group("subset")
        if subset is None:
            subset = default_subset
        split = m.group("split")
        if split is None:
            split = default_split
        num_samples = int(m.group("num_samples"))
        components.append(
            CompositeDatasetComponent(
                name=name,
                split=split,
                subset=subset,
                num_samples=num_samples,
            )
        )

    if not components:
        return None

    logger.info(
        f"Parsed composite dataset spec with {len(components)} components: "
        + ", ".join(
            f"{c.name}"
            f"{f'[{c.subset}]' if c.subset is not None else ''}"
            f"{f'({c.split})' if c.split is not None else ''}"
            f":{c.num_samples}"
            for c in components
        )
    )
    return components


def _load_raw_dataset(dataset_name, split, subset=None):
    """Load a raw HuggingFace dataset, handling special cases like C4."""
    try:
        if dataset_name == "allenai/c4":
            file_url = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz"
            return load_dataset(
                "json", data_files={"train": file_url}, split="train", streaming=False
            )
        else:
            load_kwargs = {}
            if subset is not None:
                load_kwargs = {"name": subset}
            return load_dataset(dataset_name, split=split, **load_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dataset '{dataset_name}' (subset={subset}, split={split}): {e}"
        )


def load_category_batches(
    dataset_name,
    split,
    subset,
    tokenizer,
    model_max_length,
    batch_size,
    split_by_category,
    return_vllm_tokens_prompt,
    truncate,
    samples_per_category,
):
    raw_ds = _load_raw_dataset(dataset_name, split, subset=subset)

    # load dataset processor
    proc_cls = DATASET_REGISTRY.get(dataset_name)
    if proc_cls is None:
        raise ValueError(
            f"No DatasetProcessor registered for '{dataset_name}'. "
            f"Supported: {list(DATASET_REGISTRY.keys())}"
        )

    # init processor & process dataset
    processor = proc_cls(
        dataset=raw_ds,
        tokenizer=tokenizer,
        max_input_len=model_max_length,
        split=split,
        split_by_category=split_by_category,
        return_vllm_tokens_prompt=return_vllm_tokens_prompt,
        truncate=truncate,
        batch_size=batch_size,
    )
    category_data_batches = processor.get_processed_dataset(
        samples_per_category=samples_per_category,
    )
    return category_data_batches


# --- Base Dataset Processors --------------------------------------------------


class BaseDatasetProcessor(ABC):
    category_field: str = "category"
    text_field: str = "text"
    completion_field: str = "completion"
    prompt_field: str = "prompt"
    messages_field: str = "messages"
    tools_field: str = "tools"
    all_categories_label: str = "all"

    def __init__(
        self,
        dataset: Dataset | DatasetDict,
        tokenizer: AutoTokenizer,
        pack_samples: bool = True,
        max_input_len: int | None = None,
        split: str | None = None,
        split_by_category: bool = True,
        return_vllm_tokens_prompt: bool = False,
        truncate: bool = False,
        select_only_categories: list[str] | str | None = None,
        batch_size: int = 1,
    ):
        """Defines base functionality for all Dataset Processors.

        Args:
            dataset (Dataset | DatasetDict): _description_
            tokenizer (AutoTokenizer): _description_
            split (str | None, optional): _description_. Defaults to None.
            split_by_category (bool, optional): _description_. Defaults to True.
            return_vllm_tokens_prompt (bool, optional): If True, will return
                TokensPrompt objects instead of BatchEncoding. Defaults to False
            truncate (bool, optional): If True, will truncate the samples from
                the dataset to the max_input_len instead of skipping them.
            batch_size (int, optional): Number of samples per batch. Defaults to 1.

        """
        if isinstance(dataset, DatasetDict):
            if split is None:
                split = list(dataset.keys())[0]
                logging.warning(
                    f"Using split '{split}' as default for dataset. Available "
                    f"splits: {list(dataset.keys())}",
                )
            dataset = dataset[split]
        if max_input_len is None:
            max_input_len = tokenizer.model_max_length
            logger.warning(
                f"max_input_len is set to {max_input_len} as per tokenizer's "
                f"model_max_length. This will be used for truncation.",
            )
        self.pack_samples = pack_samples
        self.max_input_len = max_input_len
        self.dataset = dataset
        self._mapped_dataset = None
        self.tokenizer = tokenizer
        self.split_by_category = split_by_category
        self.return_vllm_tokens_prompt = return_vllm_tokens_prompt
        self.truncate = truncate
        self.categories = self.get_categories()
        if isinstance(select_only_categories, str):
            select_only_categories = [select_only_categories]
        self.select_only_categories = select_only_categories
        self.batch_size = batch_size
        if self.select_only_categories:
            logger.warning(
                "select_only_categories is not None but split_by_category "
                "was False. Setting split_by_category to True and processing "
                f"categories: {self.select_only_categories}"
            )
            self.split_by_category = True
            if self.category_field not in self.dataset.column_names:
                raise RuntimeError(
                    f"Category field '{self.category_field}' not found in dataset. "
                    "Please ensure the dataset has a category field.",
                )
            for category in self.select_only_categories:
                if category not in self.categories:
                    raise RuntimeError(
                        f"Category '{category}' not found in dataset. "
                        "Please ensure the dataset has the specified categories.",
                    )

    @staticmethod
    @abstractmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        """Map a row of the dataset to the desired output format.

        EG., map "prompts" and "completions" to "messages" for chat datasets.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses.",
        )

    @abstractmethod
    def _encode_sample(self, sample: dict) -> torch.Tensor:
        """Encode a sample from the desired category of the dataset into
        tokens.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses.",
        )

    def get_processed_dataset(
        self, samples_per_category: int
    ) -> dict[str, list[TokensPrompt]] | dict[str, list[BatchEncoding]]:
        """Get requests for each category in the dataset."""
        if self._mapped_dataset is None:
            self._mapped_dataset = self.dataset.map(self._map_fn)
        if self.split_by_category:
            categories = (
                self.categories
                if self.select_only_categories is None
                else self.select_only_categories
            )
            return {
                c: self._process_samples_for_category(c, samples_per_category)
                for c in categories
            }
        else:
            return {
                self.all_categories_label: self._process_samples_for_category(
                    self.all_categories_label, samples_per_category
                ),
            }

    def get_categories(self) -> list[str]:
        """Get the unique categories in the dataset."""
        if self.category_field is None:
            logger.warning(
                "No category field specified for dataset, returning 'all' category."
            )
            return ["all"]
        return self.dataset.unique(self.category_field)

    def _process_samples_for_category(
        self,
        category: str,
        samples_per_category: int,
    ) -> list[TokensPrompt] | list[BatchEncoding]:
        if category != self.all_categories_label:
            category_dataset = self._mapped_dataset.filter(
                lambda sample: sample[self.category_field] == category,
            )
        else:
            category_dataset = self._mapped_dataset
            category = self.all_categories_label

        if self.pack_samples:
            return self._process_samples_for_category_packed(
                category, samples_per_category, category_dataset
            )
        else:
            return self._process_samples_for_category_unpacked(
                category, samples_per_category, category_dataset
            )

    def _collate_batch(self, batch: list[dict[str, torch.Tensor]]) -> BatchEncoding:
        """Collate a list of tokenized samples into a padded batch.

        Args:
            batch: List of dicts, each containing:
                - "input_ids": Tensor with shape (1, seq_len)
                - "attention_mask": Tensor with shape (1, seq_len)

        Returns:
            BatchEncoding with:
                - input_ids: Tensor of shape (batch_size, max_seq_len)
                - attention_mask: Tensor of shape (batch_size, max_seq_len)
        """
        max_len = max(sample["input_ids"].shape[-1] for sample in batch)
        pad_token_id = self.tokenizer.pad_token_id or 0

        padded_input_ids = []
        padded_attention_masks = []

        for sample in batch:
            input_ids = sample["input_ids"]
            attention_mask = sample["attention_mask"]

            seq_len = input_ids.shape[-1]
            padding_len = max_len - seq_len

            if padding_len > 0:
                input_padding = torch.full(
                    (input_ids.shape[0], padding_len),
                    pad_token_id,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                mask_padding = torch.zeros(
                    (attention_mask.shape[0], padding_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                input_ids = torch.cat([input_ids, input_padding], dim=-1)
                attention_mask = torch.cat([attention_mask, mask_padding], dim=-1)

            padded_input_ids.append(input_ids)
            padded_attention_masks.append(attention_mask)

        return BatchEncoding(
            {
                "input_ids": torch.cat(padded_input_ids, dim=0),
                "attention_mask": torch.cat(padded_attention_masks, dim=0),
            }
        )

    def _process_samples_for_category_unpacked(
        self,
        category: str,
        samples_per_category: int,
        category_dataset: Dataset,
    ) -> list[TokensPrompt] | list[BatchEncoding]:
        processed_samples = []
        sampled = []  # sample without replacement
        current_batch = []
        while len(processed_samples) < samples_per_category:
            if len(sampled) >= len(category_dataset):
                logger.warning(
                    f"Not enough samples in category '{category}' to reach "
                    f"{samples_per_category} samples. Only {len(sampled)} "
                    "samples were processed.",
                )
                break
            sample_idx = random.randint(0, len(category_dataset) - 1)
            if sample_idx in sampled:
                continue
            sampled.append(sample_idx)
            sample = category_dataset[sample_idx]
            encoded_sample = self._encode_sample(sample)
            if encoded_sample.shape[-1] > self.max_input_len:
                if self.truncate:
                    encoded_sample = encoded_sample[:, : self.max_input_len]
                else:
                    continue
            attention_mask = torch.ones((1, encoded_sample.shape[-1]), dtype=torch.long)

            if self.return_vllm_tokens_prompt:
                encoded_sample = TokensPrompt(
                    prompt_token_ids=encoded_sample[0, :-1].tolist()
                )
                processed_samples.append(encoded_sample)
            else:
                current_batch.append(
                    {"input_ids": encoded_sample, "attention_mask": attention_mask}
                )
                if len(current_batch) >= self.batch_size:
                    batched = self._collate_batch(current_batch)
                    processed_samples.append(batched)
                    current_batch = []

        # handle remaining samples in the last batch
        if current_batch and not self.return_vllm_tokens_prompt:
            batched = self._collate_batch(current_batch)
            processed_samples.append(batched)

        return processed_samples

    def _process_samples_for_category_packed(
        self,
        category: str,
        samples_per_category: int,
        category_dataset: Dataset,
    ) -> list[TokensPrompt] | list[BatchEncoding]:
        processed_samples = []
        sampled = []
        current_batch = []
        while len(processed_samples) < samples_per_category:
            if len(sampled) >= len(category_dataset):
                logger.warning(
                    f"Not enough samples in category '{category}' to reach "
                    f"{samples_per_category} samples. Only {len(sampled)} "
                    "samples were processed.",
                )
                break
            seq = torch.zeros((1, self.max_input_len), dtype=torch.long)
            seq_idx = 0
            attention_mask = torch.zeros((1, self.max_input_len), dtype=torch.long)
            while seq_idx < self.max_input_len:
                if len(sampled) >= len(category_dataset):
                    logger.warning(
                        f"Not enough samples to pack last sequence to max_input_len."
                    )
                    break
                sample_idx = random.randint(0, len(category_dataset) - 1)
                if sample_idx in sampled:
                    continue
                sampled.append(sample_idx)
                sample = category_dataset[sample_idx]
                encoded_sample = self._encode_sample(sample)  # shape (batch, seq)
                end_seq = seq_idx + encoded_sample.shape[-1]
                if end_seq > self.max_input_len:
                    encoded_sample = encoded_sample[:, : (self.max_input_len - seq_idx)]
                    end_seq = self.max_input_len
                seq[:, seq_idx:end_seq] = encoded_sample
                attention_mask[:, seq_idx:end_seq] = 1
                seq_idx = end_seq + 1
            if self.return_vllm_tokens_prompt:
                encoded_sample = TokensPrompt(
                    prompt_token_ids=seq[0, :-1].tolist()  # -1 for vLLM.LLM.generate
                )
                processed_samples.append(encoded_sample)
            else:
                current_batch.append(
                    {"input_ids": seq, "attention_mask": attention_mask}
                )
                if len(current_batch) >= self.batch_size:
                    batched = self._collate_batch(current_batch)
                    processed_samples.append(batched)
                    current_batch = []

        # handle remaining samples in the last batch
        if current_batch and not self.return_vllm_tokens_prompt:
            batched = self._collate_batch(current_batch)
            processed_samples.append(batched)

        return processed_samples


class ChatDatasetProcessor(BaseDatasetProcessor):
    def _encode_sample(self, sample: dict) -> torch.Tensor:
        chat_template_kwargs = {}
        if self.tools_field in sample:
            chat_template_kwargs = {"tools": _maybe_json_load(sample[self.tools_field])}
        chat_sample = self.tokenizer.apply_chat_template(
            sample[self.messages_field],
            add_generation_prompt=False,
            tokenize=False,
            **chat_template_kwargs,
        )
        return self.tokenizer(
            chat_sample,
            truncation=self.truncate,
            max_length=self.tokenizer.model_max_length if self.truncate else None,
            return_tensors="pt",
        )["input_ids"]

    def get_llmcompressor_dataset(self) -> Dataset:
        """Get the mapped dataset without tokenization applied."""

        def chat_template_fn(sample: dict[str, any]) -> dict[str, any]:
            """Apply chat template to the sample."""
            chat_template_kwargs = {}
            if self.tools_field in sample:
                chat_template_kwargs = {
                    "tools": _maybe_json_load(sample[self.tools_field])
                }
            chat_sample = self.tokenizer.apply_chat_template(
                sample[self.messages_field],
                add_generation_prompt=False,
                tokenize=False,
                **chat_template_kwargs,
            )
            return {"text": chat_sample}

        if self._mapped_dataset is None:
            self._mapped_dataset = self.dataset.map(self._map_fn)

        return self._mapped_dataset.map(chat_template_fn)


class LMDatasetProcessor(BaseDatasetProcessor):
    def _encode_sample(self, sample: str) -> torch.Tensor:
        return self.tokenizer(
            sample[self.text_field],
            truncation=self.truncate,
            max_length=self.tokenizer.model_max_length if self.truncate else None,
            return_tensors="pt",
        )["input_ids"]

    def get_llmcompressor_dataset(self) -> Dataset:
        """Get the mapped dataset without tokenization applied."""

        if self._mapped_dataset is None:
            self._mapped_dataset = self.dataset.map(self._map_fn)

        return self._mapped_dataset


### --- Concrete Implementations -----------------------------------------------


class CodeFeedbackChatDataset(ChatDatasetProcessor):
    category_field: str = "lang"
    messages_field: str = "text_fieldmessages"

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return sample


class TuluSFTMixtureChatDataset(ChatDatasetProcessor):
    category_field: str = "source"

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return sample


class PersonasMathChatDataset(ChatDatasetProcessor):
    """Dataset for Tulu-3 SFT Personas Math."""

    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return sample


class WildChatSFTMixtureChatDataset(ChatDatasetProcessor):
    category_field: str = "langauge"

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return sample


class MmluChatDataset(ChatDatasetProcessor):
    category_field: str = "subject"

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"{sample['question']} "
                        f"Choose from the following options: {sample['choices']}"
                    ),
                },
            ],
        }


class MagicoderEvolInstructChatDataset(ChatDatasetProcessor):
    """Dataset for Magicoder-Evol-Instruct-110K."""

    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return {
            "messages": [
                {"role": "user", "content": sample["instruction"]},
                {"role": "assistant", "content": sample["response"]},
            ],
        }


class C4LMDataset(LMDatasetProcessor):
    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return sample


class CodeAlpacaChatDataset(ChatDatasetProcessor):
    """Dataset for evol-codealpaca-v1."""

    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return {
            "messages": [
                {"role": "user", "content": sample["instruction"]},
                {"role": "assistant", "content": sample["output"]},
            ],
        }


class WritingPromptsChatDataset(ChatDatasetProcessor):
    """Dataset for WritingPrompts_curated."""

    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"Please write a creative story using the following writing prompt:\n\n {sample['prompt']}",
                },
                {"role": "assistant", "content": sample["body"]},
            ],
        }


class MixtureOfThoughtsDataset(ChatDatasetProcessor):
    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return sample


class XLamFunctionCallingDataset(ChatDatasetProcessor):
    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        tool_calls = []
        gt_tool_calls = _maybe_json_load(sample["answers"])

        for tool_call in gt_tool_calls:
            tool_calls.append(
                {
                    "function": {
                        "arguments": json.dumps(tool_call["arguments"]),
                        "name": tool_call["name"],
                    },
                    "id": f"chatcmpl-tool-{uuid.uuid4()}",
                    "type": "function",
                }
            )

        return {
            "messages": [
                {"role": "user", "content": sample["query"]},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls,
                },
            ],
            "tools": (
                sample["tools"]
                if isinstance(sample["tools"], str)
                else json.dumps(sample["tools"])
            ),
        }


class SWESmithTrajectoriesDataset(ChatDatasetProcessor):
    category_field: str = None

    tools = [
        {
            "type": "function",
            "function": {
                "name": "str_replace_editor",
                "description": (
                    "Custom editing tool for viewing, creating and editing files.\n"
                    "State is persistent across calls. If `path` is a file, `view` shows `cat -n`; "
                    "if a directory, `view` lists non-hidden entries up to 2 levels. "
                    "The `create` command fails if `path` already exists as a file. "
                    "Long outputs may be truncated with '<response clipped>'. "
                    "`undo_edit` reverts the last edit for `path`.\n\n"
                    "Notes for `str_replace`:\n"
                    "- `old_str` must match EXACTLY one or more consecutive lines (watch whitespace).\n"
                    "- If `old_str` is not unique in the file, no replacement happens—include enough context.\n"
                    "- `new_str` contains the edited lines replacing `old_str`."
                ),
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to run.",
                            "enum": [
                                "view",
                                "create",
                                "str_replace",
                                "insert",
                                "undo_edit",
                            ],
                        },
                        "path": {
                            "type": "string",
                            "description": "Absolute path to file or directory, e.g. `/testbed/file.py` or `/testbed`.",
                        },
                        "file_text": {
                            "type": "string",
                            "description": "Required for `create`: the full contents of the new file.",
                        },
                        "old_str": {
                            "type": "string",
                            "description": "Required for `str_replace`: the exact string in `path` to replace.",
                        },
                        "new_str": {
                            "type": "string",
                            "description": (
                                "Optional for `str_replace` (replacement text). "
                                "Required for `insert` (the string to insert)."
                            ),
                        },
                        "insert_line": {
                            "type": "integer",
                            "description": "Required for `insert`: insert `new_str` AFTER this 1-based line number.",
                        },
                        "view_range": {
                            "type": "array",
                            "description": (
                                "Optional for `view` when `path` is a file. If omitted, shows the full file. "
                                "Provide `[start, end]` (1-based). Use `[start, -1]` to show from start to EOF."
                            ),
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                    "required": ["command", "path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "_state_anthropic",
                "description": "Internal helper to manage persistent editor state across tool calls.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit",
                "description": (
                    "Submits the current file. "
                    "Note: implementation may use internal flags (e.g., a hidden '-f') not exposed here."
                ),
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Runs the given command directly in bash.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute.",
                        }
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        formatted_messages = []
        gt_messages = _maybe_json_load(sample["messages"])
        for message in gt_messages:
            formatted_message = {
                "role": message["role"],
                "content": _normalize_message_content(message.get("content")),
            }
            if "tool_calls" in message and message["tool_calls"] is not None:
                formatted_message["tool_calls"] = []
                for tool_call in message["tool_calls"]:
                    formatted_message["tool_calls"].append(
                        {
                            "function": {
                                "arguments": tool_call["function"]["arguments"],
                                "name": tool_call["function"]["name"],
                            },
                            "id": f"chatcmpl-tool-{uuid.uuid4()}",
                            "type": "function",
                        }
                    )
            formatted_messages.append(formatted_message)

        return {
            "messages": formatted_messages,
            "tools": SWESmithTrajectoriesDataset.tools,
        }


DATASET_REGISTRY: dict[str, BaseDatasetProcessor] = {
    "m-a-p/CodeFeedback-Filtered-Instruction": CodeFeedbackChatDataset,
    "allenai/tulu-3-sft-mixture": TuluSFTMixtureChatDataset,
    "cais/mmlu": MmluChatDataset,
    "ise-uiuc/Magicoder-Evol-Instruct-110K": MagicoderEvolInstructChatDataset,
    "allenai/c4": C4LMDataset,
    "theblackcat102/evol-codealpaca-v1": CodeAlpacaChatDataset,
    "euclaise/WritingPrompts_curated": WritingPromptsChatDataset,
    "allenai/tulu-3-sft-personas-math": PersonasMathChatDataset,
    "open-r1/Mixture-of-Thoughts": MixtureOfThoughtsDataset,
    "Salesforce/xlam-function-calling-60k": XLamFunctionCallingDataset,
    "SWE-bench/SWE-smith-trajectories": SWESmithTrajectoriesDataset,
}
