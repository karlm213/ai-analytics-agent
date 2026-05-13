import anthropic
from dotenv import load_dotenv
import os

# Load your API key from .env file
load_dotenv()

# Connect to Claude
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Send your first message
message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "I am an analytics professional building an AI agent on retail data. In one paragraph, what are the most valuable insights I could surface from customer order behavior data?"}
    ]
)

# Print the response
print(message.content[0].text) 
