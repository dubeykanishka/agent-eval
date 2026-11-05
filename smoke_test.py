import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ["PROVIDER_API_KEY"])

resp = client.messages.create(
    model="claude-haiku-5-5",   # cheap tier; confirm the current name in Anthropic's docs
    max_tokens=50,
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(resp.content[0].text)
print("tokens in/out:", resp.usage.input_tokens, resp.usage.output_tokens)