import streamlit as st
import pandas as pd
import anthropic
from dotenv import load_dotenv
import os

load_dotenv()

# Page config
st.set_page_config(
    page_title="Instacart AI Analytics Agent",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0f1117;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    /* Demo notice */
    .demo-notice {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 0.75rem 1.25rem;
        border-radius: 10px;
        color: white;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1rem;
    }
    [data-testid="stMetricLabel"] {
        color: rgba(255,255,255,0.6) !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    /* Section headers */
    .section-header {
        color: rgba(255,255,255,0.5);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
    }

    /* Suggestion buttons */
    .stButton button {
        background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
        color: rgba(255,255,255,0.85) !important;
        border: 1px solid rgba(102, 126, 234, 0.4) !important;
        border-radius: 10px !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease;
        text-align: left !important;
        width: 100%;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-color: transparent !important;
        color: white !important;
        transform: translateY(-1px);
    }

/* Chat messages */
    [data-testid="stChatMessage"] {
        background: #2d3250 !important;
        border-radius: 12px !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        margin-bottom: 0.75rem !important;
        color: #f0f0f0 !important;
    }
    
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] strong {
        color: #f0f0f0 !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background: #1e2130 !important;
        border: 1px solid rgba(102, 126, 234, 0.4) !important;
        border-radius: 12px !important;
        color: white !important;
    }

    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.08) !important;
    }

    /* Powered by badge */
    .powered-by {
        text-align: center;
        color: rgba(255,255,255,0.3);
        font-size: 0.75rem;
        margin-top: 1rem;
    }

    /* Spinner */
    .stSpinner {
        color: #667eea !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🛒 Instacart AI Analytics Agent</h1>
    <p>Ask plain-English questions about the Instacart Market Basket Analysis (Kaggle dataset) — powered by Claude AI AI</p>
</div>
""", unsafe_allow_html=True)

# Demo notice
st.markdown("""
<div class="demo-notice">
    ⚡ <strong>Live Demo:</strong> Running on sample data. Full version analyzes 3.4M Instacart orders(Kaggle Data) — 
    available for live demo. Connect on <strong>LinkedIn</strong>.
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    with st.spinner("Loading data..."):
        if os.path.exists("data/orders.csv"):
            orders = pd.read_csv("data/orders.csv")
            products = pd.read_csv("data/products.csv")
            aisles = pd.read_csv("data/aisles.csv")
            departments = pd.read_csv("data/departments.csv")
            order_products = pd.read_csv("data/order_products__train.csv")
        else:
            from sample_data import get_sample_data
            orders, products, aisles, departments, order_products = get_sample_data()

        products_full = products.merge(aisles, on="aisle_id").merge(departments, on="department_id")

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

        context = f"""
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
        stats = {
            "total_orders": total_orders,
            "total_customers": total_customers,
            "total_products": total_products,
            "reorder_rate": reorder_rate,
            "avg_basket_size": avg_basket_size
        }
        return context, stats

data_context, stats = load_data()

# KPI metrics
st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Orders", f"{stats['total_orders']:,}")
col2.metric("Unique Customers", f"{stats['total_customers']:,}")
col3.metric("Total Products", f"{stats['total_products']:,}")
col4.metric("Reorder Rate", f"{stats['reorder_rate']:.1f}%")
col5.metric("Avg Basket Size", f"{stats['avg_basket_size']:.1f} items")

st.divider()

# Suggested questions
st.markdown('<div class="section-header">💡 Try Asking</div>', unsafe_allow_html=True)
cols = st.columns(3)
suggestions = [
    "🏆 Which department should we prioritize for a loyalty program?",
    "⚠️ What is our biggest churn risk and how do we fix it?",
    "🔄 Which products have the highest reorder rates?",
    "💰 If we could improve one thing to increase revenue, what would it be?",
    "🎯 Which customer segment should we target first?",
    "📅 What day of the week do customers order most?"
]
for i, suggestion in enumerate(suggestions):
    if cols[i % 3].button(suggestion, use_container_width=True):
        st.session_state.suggested = suggestion.split(" ", 1)[1]

st.divider()

# Chat section
st.markdown('<div class="section-header">💬 Ask the Agent</div>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle suggested questions
if "suggested" in st.session_state:
    prompt = st.session_state.suggested
    del st.session_state.suggested

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                system=data_context,
                messages=st.session_state.messages
            )
            answer = response.content[0].text
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

# Chat input
if prompt := st.chat_input("Ask anything about the Instacart data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                system=data_context,
                messages=st.session_state.messages
            )
            answer = response.content[0].text
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Footer
st.markdown("""
<div class="powered-by">
    Powered by Claude AI + Anthropic API · Built by Karl Mercer
</div>
""", unsafe_allow_html=True)