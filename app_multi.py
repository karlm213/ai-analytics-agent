import streamlit as st
import pandas as pd
import anthropic
from dotenv import load_dotenv
import os

load_dotenv()

# Page config
st.set_page_config(
    page_title="Multi-Source AI Analytics Agent",
    page_icon="🔄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: white; font-size: 2.2rem; font-weight: 700; margin: 0; }
    .main-header p { color: rgba(255,255,255,0.85); font-size: 1rem; margin-top: 0.5rem; }

    .source-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge-orders { background: #1a3a5c; color: #60a5fa; }
    .badge-customers { background: #1a3a2a; color: #4ade80; }
    .badge-financials { background: #3a2a1a; color: #fb923c; }
    .badge-all { background: #3a1a3a; color: #c084fc; }

    .routing-indicator {
        background: #1e2130;
        border-radius: 10px;
        padding: 8px 14px;
        font-size: 0.8rem;
        color: rgba(255,255,255,0.7);
        margin-bottom: 8px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stMetric"] {
        background: #1e2130;
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 12px;
        padding: 1rem;
    }
    [data-testid="stMetricLabel"] { color: rgba(255,255,255,0.6) !important; font-size: 0.75rem !important; text-transform: uppercase; }
    [data-testid="stMetricValue"] { color: white !important; font-size: 1.4rem !important; font-weight: 700 !important; }

    .stButton button {
        background: #1e2130;
        color: rgba(255,255,255,0.85) !important;
        border: 1px solid rgba(102,126,234,0.4) !important;
        border-radius: 10px !important;
        font-size: 0.8rem !important;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-color: transparent !important;
        color: white !important;
    }

    [data-testid="stChatMessage"] {
        background: #2d3250 !important;
        border-radius: 12px !important;
        border: 1px solid rgba(102,126,234,0.3) !important;
        margin-bottom: 0.75rem !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] strong { color: #f0f0f0 !important; }

    hr { border-color: rgba(255,255,255,0.08) !important; }
    
    .powered-by {
        text-align: center;
        color: rgba(255,255,255,0.3);
        font-size: 0.75rem;
        margin-top: 1rem;
    }
    
    .section-header {
        color: rgba(255,255,255,0.5);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🔄 Multi-Source AI Analytics Agent</h1>
    <p>Intelligently routes across Orders, Customer, and Financial data — powered by Claude AI</p>
</div>
""", unsafe_allow_html=True)

# Data source badges
st.markdown("""
<div style="text-align:center; margin-bottom:1.5rem;">
    <span class="source-badge badge-orders">📦 Orders & Products</span>
    <span class="source-badge badge-customers">👥 Customer Profiles</span>
    <span class="source-badge badge-financials">💰 Financial Performance</span>
    <span class="source-badge badge-all">🔄 Cross-Source</span>
</div>
""", unsafe_allow_html=True)

# Tool definitions
tools = [
    {
        "name": "get_orders_data",
        "description": "Use for questions about orders, products, reorder rates, basket size, shopping frequency, popular products, aisles, or departments. Use when the question is about WHAT customers are buying and HOW OFTEN.",
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"]
        }
    },
    {
        "name": "get_customer_data",
        "description": "Use for questions about customers, churn risk, membership tiers, lifetime value, regions, or customer segments. Use when the question is about WHO the customers are.",
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"]
        }
    },
    {
        "name": "get_financial_data",
        "description": "Use for questions about revenue, margins, growth, targets, or financial performance. Use when the question is about MONEY and BUSINESS PERFORMANCE.",
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"]
        }
    },
    {
        "name": "get_all_data",
        "description": "Use ONLY when the question requires combining orders, customers, AND financials together.",
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"]
        }
    }
]

system_prompt = """You are an expert retail analytics AI agent for a grocery company.
You have access to three data sources: orders/products, customer profiles, and financial performance.

ROUTING RULES:
- Products, orders, reorder rates, basket size → get_orders_data
- Customers, churn, membership, lifetime value → get_customer_data
- Revenue, margins, growth, targets → get_financial_data
- Questions combining multiple domains → get_all_data

Always call the appropriate tool FIRST before answering.
Always cite specific numbers. Be concise and actionable."""

# Load data
@st.cache_data
def load_data():
    with st.spinner("Loading all data sources..."):
        # Orders data
        if os.path.exists("data/orders.csv"):
            orders = pd.read_csv("data/orders.csv")
            products = pd.read_csv("data/products.csv")
            aisles = pd.read_csv("data/aisles.csv")
            departments_df = pd.read_csv("data/departments.csv")
            order_products = pd.read_csv("data/order_products__train.csv")
            products_full = products.merge(aisles, on="aisle_id").merge(departments_df, on="department_id")
        else:
            from sample_data import get_sample_data
            orders, products, aisles, departments_df, order_products = get_sample_data()
            products_full = products.merge(aisles, on="aisle_id").merge(departments_df, on="department_id")

        # Customer data
        if os.path.exists("data/customer_profiles.csv"):
            customers = pd.read_csv("data/customer_profiles.csv")
        else:
            from sample_data import get_sample_customers
            customers = get_sample_customers()

        # Financial data
        if os.path.exists("data/financial_performance.csv"):
            financials = pd.read_csv("data/financial_performance.csv")
        else:
            from sample_data import get_sample_financials
            financials = get_sample_financials()

        # Orders context
        orders_ctx = f"""
SOURCE: ORDERS & PRODUCTS DATA
- Total orders: {len(orders):,}
- Unique customers: {orders['user_id'].nunique():,}
- Overall reorder rate: {order_products['reordered'].mean()*100:.1f}%
- Avg basket size: {order_products.groupby('order_id')['product_id'].count().mean():.1f} items

Top 10 products:
{order_products.merge(products_full, on='product_id')['product_name'].value_counts().head(10).to_string()}

Reorder rate by department:
{order_products.merge(products_full, on='product_id').groupby('department')['reordered'].mean().sort_values(ascending=False).to_string()}
"""

        # Customer context
        if customers is not None:
            customers_ctx = f"""
SOURCE: CUSTOMER PROFILES
- Total customers: {len(customers):,}
- Avg lifetime value: ${customers['lifetime_value'].mean():,.2f}
- High churn risk: {len(customers[customers['churn_risk']=='High']):,} customers

Membership breakdown:
{customers['membership_tier'].value_counts().to_string()}

Churn risk breakdown:
{customers['churn_risk'].value_counts().to_string()}

Avg LTV by tier:
{customers.groupby('membership_tier')['lifetime_value'].mean().sort_values(ascending=False).to_string()}
"""
        else:
            customers_ctx = "Customer data not available."

        # Financial context
        if financials is not None:
            financials_ctx = f"""
SOURCE: FINANCIAL PERFORMANCE
- Total revenue: ${financials['total_revenue'].sum():,.2f}
- Avg gross margin: {financials['gross_margin_pct'].mean():.1f}%
- Avg YoY growth: {financials['yoy_growth_pct'].mean():.1f}%

Revenue by department:
{financials[['department','total_revenue']].sort_values('total_revenue', ascending=False).to_string()}

Margin by department:
{financials[['department','gross_margin_pct']].sort_values('gross_margin_pct', ascending=False).to_string()}

vs Target:
{financials[['department','vs_target_pct']].sort_values('vs_target_pct', ascending=False).to_string()}
"""
        else:
            financials_ctx = "Financial data not available."

        stats = {
            "total_orders": len(orders),
            "total_customers": len(customers) if customers is not None else orders['user_id'].nunique(),
            "high_churn": len(customers[customers['churn_risk']=='High']) if customers is not None else 0,
            "total_revenue": financials['total_revenue'].sum() if financials is not None else 0,
            "departments": len(financials) if financials is not None else 0
        }

        return orders_ctx, customers_ctx, financials_ctx, stats

orders_context, customers_context, financials_context, stats = load_data()

def handle_tool(tool_name):
    routing_map = {
        "get_orders_data": ("📦 Orders & Products", "badge-orders", orders_context),
        "get_customer_data": ("👥 Customer Profiles", "badge-customers", customers_context),
        "get_financial_data": ("💰 Financial Performance", "badge-financials", financials_context),
        "get_all_data": ("🔄 All Data Sources", "badge-all", f"{orders_context}\n\n{customers_context}\n\n{financials_context}")
    }
    return routing_map.get(tool_name, ("Unknown", "", ""))

# KPI row
st.markdown('<div class="section-header">📊 Data Overview</div>', unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Orders", f"{stats['total_orders']:,}")
col2.metric("Total Customers", f"{stats['total_customers']:,}")
col3.metric("High Churn Risk", f"{stats['high_churn']:,}")
col4.metric("Total Revenue", f"${stats['total_revenue']:,.0f}")
col5.metric("Departments", f"{stats['departments']}")

st.divider()

# Suggested questions
st.markdown('<div class="section-header">💡 Try Asking</div>', unsafe_allow_html=True)
cols = st.columns(3)
suggestions = [
    "📦 Which products have the highest reorder rates?",
    "👥 How many customers are at high churn risk?",
    "💰 Which department is missing its revenue target?",
    "🔄 Are high churn risk customers spending less in high margin departments?",
    "👥 Which membership tier has the highest lifetime value?",
    "💰 Which department has the highest profit margin?"
]
for i, s in enumerate(suggestions):
    if cols[i % 3].button(s, use_container_width=True):
        st.session_state.suggested = s.split(" ", 1)[1]

st.divider()

# Chat
st.markdown('<div class="section-header">💬 Ask the Agent</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "routing_history" not in st.session_state:
    st.session_state.routing_history = []

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and i < len(st.session_state.routing_history):
            route_label, route_badge, _ = st.session_state.routing_history[i // 2]
            st.markdown(f'<div class="routing-indicator">Routed to: <span class="source-badge {route_badge}">{route_label}</span></div>', unsafe_allow_html=True)
        st.markdown(message["content"])

def process_question(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=2048,
                system=system_prompt,
                tools=tools,
                messages=st.session_state.messages
            )

            route_label = "Unknown"
            route_badge = ""

            if response.stop_reason == "tool_use":
                tool_block = next(b for b in response.content if b.type == "tool_use")
                route_label, route_badge, data_ctx = handle_tool(tool_block.name)

                st.markdown(f'<div class="routing-indicator">Routed to: <span class="source-badge {route_badge}">{route_label}</span></div>', unsafe_allow_html=True)

                messages_with_tool = st.session_state.messages + [
                    {"role": "assistant", "content": response.content},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": data_ctx}]}
                ]

                final = client.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=2048,
                    system=system_prompt,
                    tools=tools,
                    messages=messages_with_tool
                )
                answer = final.content[0].text
            else:
                answer = response.content[0].text

            st.markdown(answer)
            st.session_state.routing_history.append((route_label, route_badge, ""))
            st.session_state.messages.append({"role": "assistant", "content": answer})

# Handle suggested
if "suggested" in st.session_state:
    prompt = st.session_state.suggested
    del st.session_state.suggested
    process_question(prompt)
    st.rerun()

# Chat input
if prompt := st.chat_input("Ask about orders, customers, or financials..."):
    process_question(prompt)

st.markdown('<div class="powered-by">Powered by Claude AI + Anthropic API · Built by Karl Mercer</div>', unsafe_allow_html=True)