import anthropic
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Your data
data = {
    "order_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "customer_id": [101, 102, 101, 103, 102, 104, 101, 103, 104, 102],
    "product": ["Milk", "Bread", "Eggs", "Milk", "Eggs", "Bread", "Milk", "Bread", "Eggs", "Milk"],
    "category": ["Dairy", "Bakery", "Dairy", "Dairy", "Dairy", "Bakery", "Dairy", "Bakery", "Dairy", "Dairy"],
    "order_value": [3.99, 2.49, 4.99, 3.99, 4.99, 2.49, 3.99, 2.49, 4.99, 3.99],
    "days_since_last_order": [0, 0, 7, 0, 14, 0, 21, 30, 7, 45]
}

df = pd.DataFrame(data)

# Build data context once — reused in every question
data_context = f"""
You are an expert retail analytics AI agent. Answer questions using ONLY this data:

Orders: {df.to_string()}

Key stats:
- Total orders: {len(df)}
- Unique customers: {df['customer_id'].nunique()}
- Total revenue: ${df['order_value'].sum():.2f}
- Revenue by category: {df.groupby('category')['order_value'].sum().to_dict()}
- Orders per customer: {df.groupby('customer_id')['order_id'].count().to_dict()}
- Average days between orders: {df['days_since_last_order'].mean():.1f}

Be specific, use the actual numbers, and keep answers concise.
"""

# Conversation memory — stores the full chat history
conversation_history = []

print("=" * 50)
print("AI Analytics Agent — Retail Data")
print("Ask any question about your customer data.")
print("Type 'quit' to exit.")
print("=" * 50)

while True:
    # Get user question
    user_input = input("\nYou: ").strip()
    
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break
    
    if not user_input:
        continue
    
    # Add question to conversation history
    conversation_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Send to Claude with full context + history
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=data_context,
        messages=conversation_history
    )
    
    # Get response
    response = message.content[0].text
    
    # Add response to history so Claude remembers it
    conversation_history.append({
        "role": "assistant", 
        "content": response
    })
    
    print(f"\nAgent: {response}") 
