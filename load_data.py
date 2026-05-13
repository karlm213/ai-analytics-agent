import anthropic
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Sample retail order data — mirrors Instacart structure
data = {
    "order_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "customer_id": [101, 102, 101, 103, 102, 104, 101, 103, 104, 102],
    "product": ["Milk", "Bread", "Eggs", "Milk", "Eggs", "Bread", "Milk", "Bread", "Eggs", "Milk"],
    "category": ["Dairy", "Bakery", "Dairy", "Dairy", "Dairy", "Bakery", "Dairy", "Bakery", "Dairy", "Dairy"],
    "order_value": [3.99, 2.49, 4.99, 3.99, 4.99, 2.49, 3.99, 2.49, 4.99, 3.99],
    "days_since_last_order": [0, 0, 7, 0, 14, 0, 21, 30, 7, 45]
}

df = pd.DataFrame(data)

# Get summary stats to feed to Claude
summary = f"""
Customer Order Data Summary:
- Total orders: {len(df)}
- Unique customers: {df['customer_id'].nunique()}
- Total revenue: ${df['order_value'].sum():.2f}
- Most ordered product: {df['product'].value_counts().index[0]}
- Average order value: ${df['order_value'].mean():.2f}
- Average days between orders: {df['days_since_last_order'].mean():.1f}

Customer purchase frequency:
{df.groupby('customer_id')['order_id'].count().to_string()}

Revenue by category:
{df.groupby('category')['order_value'].sum().to_string()}
"""

print("Data loaded! Here's what we have:")
print(summary)
print("\nAsking Claude to analyze this data...\n")

# Ask Claude to analyze the data
message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": f"You are an expert retail analytics AI. Analyze this customer order data and give me 3 specific, actionable business insights:\n\n{summary}"
        }
    ]
)

print("Claude's Analysis:")
print(message.content[0].text)
