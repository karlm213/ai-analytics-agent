import pandas as pd
import numpy as np

np.random.seed(42)

# Products with realistic Instacart-like data
products_data = {
    "product_id": range(1, 51),
    "product_name": [
        "Organic Banana", "Bag of Organic Bananas", "Organic Strawberries",
        "Organic Baby Spinach", "Large Lemon", "Organic Avocado",
        "Organic Whole Milk", "Organic Half & Half", "Grade AA Eggs",
        "Organic Large Extra Large Eggs", "Organic Raspberry",
        "Sparkling Water Grapefruit", "Sparkling Water Lime",
        "Organic Cucumber", "Organic Garlic", "Organic Yellow Onion",
        "Organic Grape Tomatoes", "Organic Fuji Apple", "Organic Blueberries",
        "Organic Whole String Cheese", "Sourdough Bread", "Wheat Bread",
        "Organic Orange Juice", "Almond Milk", "Greek Yogurt",
        "Organic Butter", "Cheddar Cheese", "Organic Chicken Breast",
        "Organic Ground Beef", "Atlantic Salmon",
        "Sparkling Water Blackberry", "Organic Apple Juice",
        "Organic Carrots", "Broccoli Crown", "Organic Celery",
        "Organic Romaine Lettuce", "Organic Kale", "Organic Mushrooms",
        "Organic Red Pepper", "Organic Zucchini",
        "Chocolate Chip Cookies", "Organic Tortilla Chips",
        "Sparkling Water Plain", "Organic Coffee", "Green Tea",
        "Organic Pasta", "Organic Tomato Sauce", "Brown Rice",
        "Organic Olive Oil", "Organic Honey"
    ],
    "aisle_id": [
        24, 24, 24, 24, 24, 24,  # Produce
        84, 84, 86, 86,          # Dairy
        24, 115, 115,            # Produce/Beverages
        24, 24, 24, 24, 24, 24, 21,  # Produce/Cheese
        112, 112,                # Bread
        98, 91, 84,              # Beverages/Dairy
        84, 21, 49, 49, 49,      # Dairy/Meat
        115, 98,                 # Beverages
        24, 24, 24, 24, 24, 24, 24, 24,  # Produce
        19, 107,                 # Snacks
        115, 57, 57,             # Beverages/Tea
        9, 9, 9, 9, 9            # Dry Goods
    ],
    "department_id": [
        4, 4, 4, 4, 4, 4,        # Produce
        16, 16, 16, 16,          # Dairy
        4, 7, 7,                 # Produce/Beverages
        4, 4, 4, 4, 4, 4, 16,   # Produce/Dairy
        3, 3,                    # Bakery
        7, 7, 16,                # Beverages/Dairy
        16, 16, 12, 12, 12,      # Dairy/Meat
        7, 7,                    # Beverages
        4, 4, 4, 4, 4, 4, 4, 4, # Produce
        19, 19,                  # Snacks
        7, 7, 7,                 # Beverages
        9, 9, 9, 9, 9            # Dry Goods
    ]
}

aisles_data = {
    "aisle_id": [24, 84, 86, 115, 21, 112, 98, 91, 49, 19, 107, 57, 9, 3, 7, 16],
    "aisle": [
        "fresh fruits", "milk", "eggs", "water seltzer sparkling water",
        "packaged cheese", "bread", "juice nectars", "yogurt",
        "meat counter", "snacks", "cookies cakes", "tea",
        "dry pasta", "fresh pasta", "condiments", "butter"
    ]
}

departments_data = {
    "department_id": [4, 16, 7, 3, 12, 19, 9],
    "department": [
        "produce", "dairy eggs", "beverages",
        "bakery", "meat seafood", "snacks", "dry goods"
    ]
}

# Generate 1000 realistic orders
n_orders = 1000
n_customers = 200

orders_data = {
    "order_id": range(1, n_orders + 1),
    "user_id": np.random.randint(1, n_customers + 1, n_orders),
    "order_number": np.random.randint(1, 20, n_orders),
    "order_dow": np.random.choice([0,1,2,3,4,5,6], n_orders,
                  p=[0.11, 0.14, 0.16, 0.15, 0.14, 0.16, 0.14]),
"order_hour_of_day": np.random.choice(range(24), n_orders,
                      p=[0.01,0.01,0.01,0.01,0.01,0.02,0.03,0.06,
                         0.09,0.10,0.10,0.09,0.08,0.07,0.06,0.05,
                         0.05,0.04,0.04,0.03,0.02,0.01,0.01,0.02]),
    "days_since_prior_order": np.random.choice(
        [1,2,3,4,5,6,7,8,9,10,14,21,30],
        n_orders,
        p=[0.03,0.03,0.04,0.04,0.05,0.05,0.20,0.05,0.04,0.04,0.10,0.08,0.25]
    )
}

# Generate order products — each order has 3-15 items
order_products_list = []
order_id_counter = 1

for order_id in range(1, n_orders + 1):
    n_items = np.random.randint(3, 16)
    selected_products = np.random.choice(range(1, 51), n_items, replace=False)

    for i, product_id in enumerate(selected_products):
        # Higher reorder rate for staples (products 1-20)
        if product_id <= 20:
            reordered = np.random.choice([0, 1], p=[0.25, 0.75])
        else:
            reordered = np.random.choice([0, 1], p=[0.55, 0.45])

        order_products_list.append({
            "order_id": order_id,
            "product_id": product_id,
            "add_to_cart_order": i + 1,
            "reordered": reordered
        })

# Create DataFrames
orders = pd.DataFrame(orders_data)
products = pd.DataFrame(products_data)
aisles = pd.DataFrame(aisles_data)
departments = pd.DataFrame(departments_data)
order_products = pd.DataFrame(order_products_list)

def get_sample_data():
    return orders, products, aisles, departments, order_products