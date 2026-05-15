# AI Analytics Agent 🤖

A conversational AI agent that lets you ask plain-English questions 
about retail data and get instant, data-driven insights.

## What It Does

Instead of writing SQL queries or building dashboards, just ask:

- *"Which customer is most at risk of churning?"*
- *"What product should I promote this week?"*
- *"Who is my most valuable customer and why?"*

The agent analyzes the data and responds with specific, 
actionable insights in seconds.

## Why I Built This

I'm passionate about emerging technology — specifically how AI is 
changing the way people interact with data. This project is where 
I practice and build those skills hands-on.

My goal is to explore how AI agents can replace manual reporting 
and make data insights accessible to anyone, not just analysts.

## Tech Stack

- **Python** — core language
- **Anthropic Claude API** — LLM backbone
- **Pandas** — data loading and transformation
- **LangChain** — agent orchestration (coming soon)
- **ChromaDB** — vector database for RAG (coming soon)
- **Streamlit** — web UI (coming soon)

## How To Run It

1. Clone this repo
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate`
4. Install dependencies: `pip install anthropic pandas python-dotenv`
5. Create a `.env` file and add your Anthropic API key:

6. Run the agent: `python agent.py`


## Data Setup

This project uses the Instacart Market Basket Analysis dataset from Kaggle.

**To run this project locally:**
1. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/instacart-market-basket-analysis/data)
2. Create a `/data` folder in the project root
3. Place these 5 CSV files inside it:
   - `orders.csv`
   - `order_products__train.csv`
   - `products.csv`
   - `aisles.csv`
   - `departments.csv`

> **Note:** The data folder is excluded from this repo via `.gitignore` 
> due to file size and Kaggle's redistribution terms. 
> You must download it directly from Kaggle (free account required).
## Let's Connect

Always happy to talk AI, analytics, and emerging tech.
Find me at linkedin.com/in/karlmercer/


| App | Description | Link |
|-----|-------------|------|
| 🛒 Single Source Agent | AI analytics on retail order data | [Launch App](https://ai-analytics-agent-retail.streamlit.app) |
| 🔄 Multi-Source Agent | Intelligent routing across Orders, Customers & Financials | [Launch App](https://ai-analytics-agent-retail-multiple-sources.streamlit.app) |
