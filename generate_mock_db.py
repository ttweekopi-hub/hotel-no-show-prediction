"""
This module generates a mock SQLite database to simulate the hotel's raw reservation records.

I created this script to generate synthetic booking data containing real-world data anomalies
such as missing values, negative days, and mixed currencies. This database serves as a starting
point to test my entire machine learning pipeline in a clean, reproducible way.
"""

import os
import sqlite3
import numpy as np
import pandas as pd

def generate_mock_db(db_path: str = "data/noshow.db", n_rows: int = 500):
    """Generates a mock SQLite database representing the hotel reservation records.
    
    This synthetic dataset mirrors the real schema and data quality anomalies,
    providing a complete integration test dataset for the cleaning, feature
    engineering, and ML modeling phases of the pipeline.
    """
    print(f"Initializing mock database generation at: {db_path}")
    
    # Ensure parent data directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Delete the database if it already exists to ensure a clean slate
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database at {db_path} to reset.")

    # Set random seed for complete reproducibility
    np.random.seed(42)
    
    # Column 1: booking_id (sequential integers)
    booking_ids = np.arange(10000, 10000 + n_rows)
    
    # Column 2: no_show (target variable: float 0.0 or 1.0, and 5% missing to test dropping null targets)
    no_show = np.random.choice([0.0, 1.0, None], size=n_rows, p=[0.65, 0.30, 0.05])
    
    # Column 3: branch (categorical branches: 'Changi' or 'Orchard')
    branch = np.random.choice(["Changi", "Orchard"], size=n_rows)
    
    # Column 4, 5, 7: booking_month, arrival_month, checkout_month
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    
    arrival_month = np.random.choice(months, size=n_rows)
    booking_month = np.random.choice(months, size=n_rows)
    checkout_month = arrival_month.copy()  # Simulating short stays within the same month
    
    # Introduce varied month casing to test clean.py normalization (e.g. MaY, jUne)
    casing_variants = [lambda m: m, lambda m: m.upper(), lambda m: m.lower(), lambda m: m[0].lower() + m[1:].upper()]
    for i in range(n_rows):
        if np.random.rand() < 0.15:
            variant_fn = np.random.choice(casing_variants)
            arrival_month[i] = variant_fn(arrival_month[i])
            booking_month[i] = variant_fn(booking_month[i])
            checkout_month[i] = variant_fn(checkout_month[i])
            
    # Column 6 & 8: arrival_day & checkout_day
    arrival_day = np.random.randint(1, 28, size=n_rows).astype(float)
    # Add random negative days (5% probability) to verify absolute conversion in clean.py
    neg_idx = np.random.choice(n_rows, size=int(0.05 * n_rows), replace=False)
    arrival_day[neg_idx] = -arrival_day[neg_idx]
    
    # Checkout day is usually arrival_day + a few days
    checkout_day = np.abs(arrival_day) + np.random.randint(1, 4, size=n_rows)
    # Add random negative checkout days too
    neg_checkout_idx = np.random.choice(n_rows, size=int(0.02 * n_rows), replace=False)
    checkout_day[neg_checkout_idx] = -checkout_day[neg_checkout_idx]
    
    # Column 9: country (categorical)
    country = np.random.choice(["Singapore", "Malaysia", "Indonesia", "Australia", "China"], size=n_rows)
    
    # Column 10: first_time (binary guest loyalty status: 'Yes' or 'No')
    first_time = np.random.choice(["Yes", "No"], size=n_rows)
    
    # Column 11: room (room type with 5% missing to test the dynamic KNN Room Imputer)
    room = np.random.choice(["Single", "Queen", "King", "President Suite", None], size=n_rows, p=[0.40, 0.30, 0.20, 0.05, 0.05])
    
    # Column 12: price (mixed currencies e.g. SGD$, USD, and 5% missing to test dynamic Random Forest Price Imputer)
    price_bases = {"Single": 150, "Queen": 250, "King": 450, "President Suite": 1200}
    price = []
    for r in room:
        if r is None or np.random.rand() < 0.05:
            price.append(None)
        else:
            base = price_bases[r]
            # Vary currencies (SGD$ or USD) and construct pricing text columns
            currency = np.random.choice(["SGD$ ", "USD "])
            val = base + np.random.normal(0, 20)
            price.append(f"{currency}{val:.2f}")
            
    # Column 13: platform (booking source)
    platform = np.random.choice(["Website", "Mobile App", "Agency"], size=n_rows)
    
    # Column 14: num_adults (represented as string numbers or text words to test mapper in clean.py)
    num_adults = np.random.choice(["1", "2", "one", "two"], size=n_rows, p=[0.45, 0.45, 0.05, 0.05])
    
    # Column 15: num_children (floats)
    num_children = np.random.choice([0.0, 1.0, 2.0], size=n_rows, p=[0.85, 0.10, 0.05])
    
    # Create pandas DataFrame
    df = pd.DataFrame({
        "booking_id": booking_ids,
        "no_show": no_show,
        "branch": branch,
        "booking_month": booking_month,
        "arrival_month": arrival_month,
        "arrival_day": arrival_day,
        "checkout_month": checkout_month,
        "checkout_day": checkout_day,
        "country": country,
        "first_time": first_time,
        "room": room,
        "price": price,
        "platform": platform,
        "num_adults": num_adults,
        "num_children": num_children
    })
    
    # Write to SQLite
    conn = sqlite3.connect(db_path)
    df.to_sql("noshow", conn, if_exists="replace", index=False)
    conn.close()
    
    print(f"Successfully generated mock database at {db_path} containing {n_rows} rows.")
    print("Database table details:")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    generate_mock_db()
