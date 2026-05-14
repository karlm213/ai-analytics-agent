import pandas as pd
import anthropic
from dotenv import load_dotenv
import os

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

print("Loading Instacart data... (this takes about 20 seconds)")

# Load all 5 files
orders = pd.read_csv("data/orders.csv")
products = pd.read_csv("data/products.csv")
aisles = pd.read_csv("data/aisles.csv")
departments = pd.read_csv("data/departments.csv")
order_products = pd.read_csv("data/order_products__train.csv")

# Merge products with aisles and departments
products_full = products.merge(aisles, on="aisle_id").merge(departments, on="department_id")

# Pre-calculate key stats once at startup
total_orders = len(orders)
total_customers = orders["user_id"].nunique()
total_products = len(products)
reorder_rate = order_products["reordered"].mean() * 100
avg_basket_size = order_products.groupby("order_id")["product_id"].count().mean()
top_products = order_products.merge(products_full, on="product_id")["product_name"].value_counts().head(20)
top_aisles = order_products.merge(products_full, on="product_id")["aisle"].value_counts().head(10)
top_departments = order_products.merge(products_full, on="product_id")["department"].value_counts().head(10)
order_frequency = orders["days_since_prior_order"].value_counts().head(10)
reorder_by_department = order_products.merge(products_full, on="product_id").groupby("department")["reordered"].mean().sort_values(ascending=False)

# Build rich data context
data_context = f"""
You are an expert retail analytics AI agent with deep knowledge of the Instacart dataset.
Answer every question using ONLY the data provided below. Always cite specific numbers.
Be concise, specific, and actionable. Think like a senior retail analyst presenting to a VP.

DATASET OVERVIEW:
- Total orders analyzed: {total_orders:,}
- Total unique customers: {total_customers:,}
- Total products available: {total_products:,}
- Overall reorder rate: {reorder_rate:.1f}%
- Average basket size: {avg_basket_size:.1f} items per order

TOP 20 MOST ORDERED PRODUCTS:
{top_products.to_string()}

TOP 10 MOST POPULAR AISLES:
{top_aisles.to_string()}

TOP 10 DEPARTMENTS BY ORDER VOLUME:
{top_departments.to_string()}

ORDER FREQUENCY PATTERNS (days since prior order):
{order_frequency.to_string()}

REORDER RATE BY DEPARTMENT:
{reorder_by_department.to_string()}
"""

# Conversation memory
conversation_history = []

print("\n" + "="*60)
print("   Instacart AI Analytics Agent")
print("   Powered by Claude + 3.4M real orders")
print("="*60)
print("\nData loaded! Ask me anything about the Instacart dataset.")
print("Type 'quit' to exit | Type 'reset' to start a new conversation")
print("="*60)

while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() == 'quit':
        print("\nGoodbye!")
        break

    if user_input.lower() == 'reset':
        conversation_history = []
        print("\nConversation reset. Ask me anything!")
        continue

    if not user_input:
        continue

    # Add question to history
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

    response = message.content[0].text

    # Add response to history
    conversation_history.append({
        "role": "assistant",
        "content": response
    })

    print(f"\nAgent: {response}") 
