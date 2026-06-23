# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest

from vllm.benchmarks.datasets import CustomDataset


class TokenizerShouldNotBeCalled:

    def __call__(self, prompt):
        raise AssertionError(f"token-id prompt was tokenized: {prompt!r}")

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("chat template was applied to a token-id prompt")


@pytest.mark.benchmark
def test_custom_dataset_accepts_token_id_prompts(tmp_path) -> None:
    pytest.importorskip("pandas")
    dataset_path = tmp_path / "token_ids.jsonl"
    records = [
        {"prompt_token_ids": [1, 2, 3], "output_tokens": 4},
        {"input_ids": [5, 6], "output_tokens": 7},
        {"token_ids": [8], "output_tokens": 9},
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    dataset = CustomDataset(dataset_path=str(dataset_path), disable_shuffle=True)
    requests = dataset.sample(
        tokenizer=TokenizerShouldNotBeCalled(),
        num_requests=0,
        output_len=-1,
        skip_chat_template=False,
    )

    assert [request.prompt for request in requests] == [[1, 2, 3], [5, 6], [8]]
    assert [request.prompt_len for request in requests] == [3, 2, 1]
    assert [request.expected_output_len for request in requests] == [4, 7, 9]


@pytest.mark.benchmark
def test_custom_dataset_rejects_invalid_token_id_prompt(tmp_path) -> None:
    pytest.importorskip("pandas")
    dataset_path = tmp_path / "invalid_token_ids.jsonl"
    dataset_path.write_text(
        json.dumps({"prompt_token_ids": [1, True, 3], "output_tokens": 4}),
        encoding="utf-8",
    )

    dataset = CustomDataset(dataset_path=str(dataset_path), disable_shuffle=True)
    with pytest.raises(ValueError, match="prompt_token_ids\\[1\\]"):
        dataset.sample(
            tokenizer=TokenizerShouldNotBeCalled(),
            num_requests=0,
            output_len=-1,
        )
