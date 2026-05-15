import pandas as pd
import anthropic
from dotenv import load_dotenv
import os

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

print("Loading all data sources...")

# ---- Load all 3 data sources ----
# Source 1: Instacart Orders
orders = pd.read_csv("data/orders.csv")
products = pd.read_csv("data/products.csv")
aisles = pd.read_csv("data/aisles.csv")
departments = pd.read_csv("data/departments.csv")
order_products = pd.read_csv("data/order_products__train.csv")
products_full = products.merge(aisles, on="aisle_id").merge(departments, on="department_id")

# Source 2: Customer Profiles
customers = pd.read_csv("data/customer_profiles.csv")

# Source 3: Financial Performance
financials = pd.read_csv("data/financial_performance.csv")

print("✅ All 3 data sources loaded!")

# ---- Build context for each source ----
orders_context = f"""
SOURCE: ORDERS & PRODUCTS DATA
- Total orders: {len(orders):,}
- Unique customers: {orders['user_id'].nunique():,}
- Total products: {len(products):,}
- Overall reorder rate: {order_products['reordered'].mean()*100:.1f}%
- Avg basket size: {order_products.groupby('order_id')['product_id'].count().mean():.1f} items

Top 10 products by order volume:
{order_products.merge(products_full, on='product_id')['product_name'].value_counts().head(10).to_string()}

Top departments by order volume:
{order_products.merge(products_full, on='product_id')['department'].value_counts().head(10).to_string()}

Reorder rate by department:
{order_products.merge(products_full, on='product_id').groupby('department')['reordered'].mean().sort_values(ascending=False).to_string()}

Order frequency (days since prior order):
{orders['days_since_prior_order'].value_counts().head(7).to_string()}
"""

customers_context = f"""
SOURCE: CUSTOMER PROFILES DATA
- Total customers: {len(customers):,}
- Average lifetime value: ${customers['lifetime_value'].mean():,.2f}
- Average order value: ${customers['avg_order_value'].mean():.2f}
- Average total orders per customer: {customers['total_orders'].mean():.1f}

Membership tier breakdown:
{customers['membership_tier'].value_counts().to_string()}

Churn risk breakdown:
{customers['churn_risk'].value_counts().to_string()}

Average lifetime value by tier:
{customers.groupby('membership_tier')['lifetime_value'].mean().sort_values(ascending=False).to_string()}

Churn risk by membership tier:
{customers.groupby(['membership_tier', 'churn_risk']).size().to_string()}

Regional breakdown:
{customers['region'].value_counts().to_string()}

Email opt-in rate: {customers['email_opt_in'].mean()*100:.1f}%

Customers at high churn risk: {len(customers[customers['churn_risk']=='High']):,}
High churn risk by tier:
{customers[customers['churn_risk']=='High']['membership_tier'].value_counts().to_string()}
"""

financials_context = f"""
SOURCE: FINANCIAL PERFORMANCE DATA
- Total annual revenue: ${financials['total_revenue'].sum():,.2f}
- Average gross margin: {financials['gross_margin_pct'].mean():.1f}%
- Average YoY growth: {financials['yoy_growth_pct'].mean():.1f}%
- Total units sold: {financials['units_sold'].sum():,}

Revenue by department:
{financials[['department','total_revenue']].sort_values('total_revenue', ascending=False).to_string()}

Gross margin by department:
{financials[['department','gross_margin_pct']].sort_values('gross_margin_pct', ascending=False).to_string()}

Performance vs target by department:
{financials[['department','vs_target_pct']].sort_values('vs_target_pct', ascending=False).to_string()}

YoY growth by department:
{financials[['department','yoy_growth_pct']].sort_values('yoy_growth_pct', ascending=False).to_string()}

Departments missing revenue target:
{financials[financials['vs_target_pct'] < 0][['department','vs_target_pct']].to_string()}
"""

# ---- Router Tool Definitions ----
tools = [
    {
        "name": "get_orders_data",
        "description": "Use this tool for questions about orders, products, reorder rates, basket size, shopping frequency, popular products, aisles, or departments. Use when the question is about WHAT customers are buying and HOW OFTEN.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question being asked about orders data"
                }
            },
            "required": ["question"]
        }
    },
    {
        "name": "get_customer_data",
        "description": "Use this tool for questions about customers, churn risk, membership tiers, lifetime value, regions, email opt-in, or customer segments. Use when the question is about WHO the customers are.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question being asked about customer data"
                }
            },
            "required": ["question"]
        }
    },
    {
        "name": "get_financial_data",
        "description": "Use this tool for questions about revenue, margins, growth, targets, financial performance, or profitability. Use when the question is about MONEY and BUSINESS PERFORMANCE.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question being asked about financial data"
                }
            },
            "required": ["question"]
        }
    },
    {
        "name": "get_all_data",
        "description": "Use this tool ONLY when the question requires combining information from orders, customers, AND financials together. Use for cross-source questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The cross-source question being asked"
                }
            },
            "required": ["question"]
        }
    }
]

def handle_tool_call(tool_name, question):
    """Returns the right data context based on which tool the agent called"""
    if tool_name == "get_orders_data":
        print(f"   📦 Routing to: Orders & Products data")
        return orders_context
    elif tool_name == "get_customer_data":
        print(f"   👥 Routing to: Customer Profiles data")
        return customers_context
    elif tool_name == "get_financial_data":
        print(f"   💰 Routing to: Financial Performance data")
        return financials_context
    elif tool_name == "get_all_data":
        print(f"   🔄 Routing to: All data sources")
        return f"{orders_context}\n\n{customers_context}\n\n{financials_context}"

# ---- System prompt ----
system_prompt = """You are an expert retail analytics AI agent for a grocery company.
You have access to three data sources: orders/products, customer profiles, and financial performance.

IMPORTANT ROUTING RULES:
- Questions about products, orders, reorder rates, basket size → use get_orders_data
- Questions about customers, churn, membership, lifetime value → use get_customer_data  
- Questions about revenue, margins, growth, targets → use get_financial_data
- Questions combining multiple domains → use get_all_data

Always call the appropriate tool FIRST before answering.
Always cite specific numbers in your answers.
Be concise and actionable — think like a senior analyst presenting to a VP."""

# ---- Agent loop ----
conversation_history = []

print("\n" + "="*60)
print("   Multi-Source AI Analytics Agent")
print("   3 Data Sources: Orders | Customers | Financials")
print("="*60)
print("\nAsk me anything about orders, customers, or financials.")
print("Type 'quit' to exit | Type 'reset' to start fresh")
print("="*60)

while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() == 'quit':
        print("\nGoodbye!")
        break

    if user_input.lower() == 'reset':
        conversation_history = []
        print("\nConversation reset!")
        continue

    if not user_input:
        continue

    conversation_history.append({
        "role": "user",
        "content": user_input
    })

    print("\n   🤔 Agent thinking...")

    # First call — agent decides which tool to use
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        system=system_prompt,
        tools=tools,
        messages=conversation_history
    )

    # Handle tool use
    if response.stop_reason == "tool_use":
        tool_use_block = next(b for b in response.content if b.type == "tool_use")
        tool_name = tool_use_block.name
        tool_input = tool_use_block.input

        # Get the right data
        data_context = handle_tool_call(tool_name, tool_input["question"])

        # Build messages with tool result
        messages_with_tool = conversation_history + [
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_block.id,
                        "content": data_context
                    }
                ]
            }
        ]

        # Second call — agent answers with the data
        final_response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2048,
            system=system_prompt,
            tools=tools,
            messages=messages_with_tool
        )

        answer = final_response.content[0].text
        conversation_history.append({
            "role": "assistant",
            "content": answer
        })

    else:
        answer = response.content[0].text
        conversation_history.append({
            "role": "assistant",
            "content": answer
        })

    print(f"\nAgent: {answer}") 
