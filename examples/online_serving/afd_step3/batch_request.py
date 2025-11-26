from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY", # Required, but can be any value
    )

    # Example for chat completions
    response = client.chat.completions.create(
        model="<your_model_name>", # Must match the model started with the server
        messages=[
            {"role": "user", "content": "Tell me a story."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        # Add other parameters like temperature, max_tokens, etc.
    )

    for choice in response.choices:
        print(choice.message.content)
