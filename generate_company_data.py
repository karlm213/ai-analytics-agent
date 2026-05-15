import pandas as pd
import numpy as np

np.random.seed(42)

# ---- DATASET 1: Customer Profiles ----
# Uses same user IDs as Instacart (1-206209)
n_customers = 1000

customer_profiles = pd.DataFrame({
    "user_id": range(1, n_customers + 1),
    "age": np.random.randint(22, 65, n_customers),
    "membership_tier": np.random.choice(
        ["Bronze", "Silver", "Gold", "Platinum"],
        n_customers,
        p=[0.40, 0.30, 0.20, 0.10]
    ),
    "lifetime_value": np.round(np.random.exponential(350, n_customers), 2),
    "days_as_member": np.random.randint(30, 1825, n_customers),
    "churn_risk": np.random.choice(
        ["Low", "Medium", "High"],
        n_customers,
        p=[0.60, 0.25, 0.15]
    ),
    "preferred_shopping_day": np.random.choice(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        n_customers
    ),
    "avg_order_value": np.round(np.random.normal(45, 15, n_customers), 2),
    "total_orders": np.random.randint(1, 50, n_customers),
    "last_order_days_ago": np.random.randint(1, 90, n_customers),
    "email_opt_in": np.random.choice([True, False], n_customers, p=[0.75, 0.25]),
    "region": np.random.choice(
        ["Northeast", "Southeast", "Midwest", "Southwest", "West"],
        n_customers,
        p=[0.25, 0.20, 0.20, 0.15, 0.20]
    )
})

# Make lifetime value correlate with membership tier
tier_multipliers = {"Bronze": 0.5, "Silver": 1.0, "Gold": 1.8, "Platinum": 3.2}
customer_profiles["lifetime_value"] = customer_profiles.apply(
    lambda x: round(x["lifetime_value"] * tier_multipliers[x["membership_tier"]], 2), axis=1
)

# Make churn risk correlate with last order days
customer_profiles.loc[customer_profiles["last_order_days_ago"] > 60, "churn_risk"] = "High"
customer_profiles.loc[customer_profiles["last_order_days_ago"] < 14, "churn_risk"] = "Low"

# ---- DATASET 2: Financial Performance ----
# Revenue by product category — ties to Instacart departments
departments = [
    "produce", "dairy eggs", "beverages", "snacks",
    "meat seafood", "bakery", "frozen", "pantry",
    "household", "personal care"
]

financial_data = pd.DataFrame({
    "department": departments,
    "q1_revenue": np.round(np.random.uniform(120000, 850000, len(departments)), 2),
    "q2_revenue": np.round(np.random.uniform(130000, 900000, len(departments)), 2),
    "q3_revenue": np.round(np.random.uniform(115000, 820000, len(departments)), 2),
    "q4_revenue": np.round(np.random.uniform(140000, 950000, len(departments)), 2),
    "revenue_target": np.round(np.random.uniform(500000, 3500000, len(departments)), 2),
    "gross_margin_pct": np.round(np.random.uniform(18, 52, len(departments)), 1),
    "yoy_growth_pct": np.round(np.random.uniform(-5, 28, len(departments)), 1),
    "avg_transaction_value": np.round(np.random.uniform(8, 45, len(departments)), 2),
    "units_sold": np.random.randint(50000, 500000, len(departments)),
    "return_rate_pct": np.round(np.random.uniform(0.5, 8.0, len(departments)), 1)
})

# Calculate total revenue and vs target
financial_data["total_revenue"] = (
    financial_data["q1_revenue"] +
    financial_data["q2_revenue"] +
    financial_data["q3_revenue"] +
    financial_data["q4_revenue"]
)
financial_data["vs_target_pct"] = np.round(
    (financial_data["total_revenue"] / financial_data["revenue_target"] - 1) * 100, 1
)

# Save to data folder
customer_profiles.to_csv("data/customer_profiles.csv", index=False)
financial_data.to_csv("data/financial_performance.csv", index=False)

print("✅ Customer profiles generated:", len(customer_profiles), "customers")
print("✅ Financial data generated:", len(financial_data), "departments")
print("\nCustomer Profile Sample:")
print(customer_profiles.head(3).to_string())
print("\nFinancial Data Sample:")
print(financial_data.head(3).to_string())
print("\n✅ Both files saved to data/ folder!") 
