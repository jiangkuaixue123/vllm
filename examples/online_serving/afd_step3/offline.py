# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm import LLM, SamplingParams

prompts = [
    "1 3 5 7 9",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(
    model="/home/models/DeepSeek-V2-Lite",
    enforce_eager=True,
    afd_config={"afd_connector":"p2pconnector", "afd_role": "attention", "num_afd_stages":"3","afd_extra_config":{"afd_size":"2A2F"}}
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"prompt{prompt!r}, generated text: {generated_text!r}")
