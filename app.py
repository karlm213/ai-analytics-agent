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

# Header
st.title("🛒 Instacart AI Analytics Agent")
st.markdown("Ask plain-English questions about 3.4 million real Instacart orders — powered by Claude AI")
st.divider()
st.info("⚡ **Live Demo Note:** This hosted version runs on a sample dataset due to cloud memory limits. The full version analyzes 3.4 million real Instacart orders and is available for live demo — contact me on LinkedIn.")

# Load data — only runs once thanks to caching
@st.cache_data
def load_data():
    with st.spinner("Loading 3.4M orders... (first load takes ~20 seconds)"):
        orders = pd.read_csv("data/orders.csv")
        products = pd.read_csv("data/products.csv")
        aisles = pd.read_csv("data/aisles.csv")
        departments = pd.read_csv("data/departments.csv")
        order_products = pd.read_csv("data/order_products__train.csv")

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

# Load data
data_context, stats = load_data()

# KPI metrics row
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Orders", f"{stats['total_orders']:,}")
col2.metric("Unique Customers", f"{stats['total_customers']:,}")
col3.metric("Total Products", f"{stats['total_products']:,}")
col4.metric("Reorder Rate", f"{stats['reorder_rate']:.1f}%")
col5.metric("Avg Basket Size", f"{stats['avg_basket_size']:.1f} items")

st.divider()

# Suggested questions
st.markdown("**💡 Try asking:**")
cols = st.columns(3)
suggestions = [
    "Which department should we prioritize for a loyalty program?",
    "What is our biggest churn risk and how do we fix it?",
    "Which products have the highest reorder rates?",
    "What day of the week do customers order most?",
    "If we could improve one thing to increase revenue, what would it be?",
    "Which customer segment should we target first?"
]
for i, suggestion in enumerate(suggestions):
    if cols[i % 3].button(suggestion, use_container_width=True):
        st.session_state.suggested = suggestion

st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle suggested question clicks
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
