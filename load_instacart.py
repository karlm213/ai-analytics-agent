import pandas as pd
import anthropic
from dotenv import load_dotenv
import os

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

print("Loading Instacart data...")

# Load all 5 files
orders = pd.read_csv("data/orders.csv")
products = pd.read_csv("data/products.csv")
aisles = pd.read_csv("data/aisles.csv")
departments = pd.read_csv("data/departments.csv")
order_products = pd.read_csv("data/order_products__train.csv")

print(f"✅ Orders loaded: {len(orders):,} rows")
print(f"✅ Products loaded: {len(products):,} rows")
print(f"✅ Aisles loaded: {len(aisles):,} rows")
print(f"✅ Departments loaded: {len(departments):,} rows")
print(f"✅ Order products loaded: {len(order_products):,} rows")

# Merge products with aisles and departments
products_full = products.merge(aisles, on="aisle_id").merge(departments, on="department_id")

# Build key stats for the agent
total_orders = len(orders)
total_customers = orders["user_id"].nunique()
total_products = len(products)
top_aisles = order_products.merge(products_full, on="product_id")["aisle"].value_counts().head(5)
top_products = order_products.merge(products_full, on="product_id")["product_name"].value_counts().head(10)
reorder_rate = order_products["reordered"].mean() * 100
avg_basket_size = order_products.groupby("order_id")["product_id"].count().mean()

# Build context summary
summary = f"""
Real Instacart Dataset Summary:
- Total orders: {total_orders:,}
- Total unique customers: {total_customers:,}
- Total products available: {total_products:,}
- Overall reorder rate: {reorder_rate:.1f}%
- Average basket size: {avg_basket_size:.1f} items per order

Top 5 most popular aisles:
{top_aisles.to_string()}

Top 10 most ordered products:
{top_products.to_string()}

Order frequency patterns:
{orders["days_since_prior_order"].value_counts().head(5).to_string()}
"""

print("\n" + "="*50)
print(summary)
print("="*50)
print("\nAsking Claude to analyze this real dataset...\n")

# Ask Claude for insights
message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": f"You are an expert retail analytics AI. Analyze this real Instacart dataset and give me 3 powerful business insights a VP of Retail would find valuable:\n\n{summary}"
        }
    ]
)

print("Claude's Analysis:")
print(message.content[0].text) 
