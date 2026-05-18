"""
This module performs quick exploratory checks on the mock database.

I use this script to inspect specific data patterns, anomalies, and value distributions
within my SQL database (such as missing values in 'no_show', price prefixes, and month names)
to ensure my cleaning logic is accurate and aligned with the actual data.
"""

import sqlite3
import pandas as pd

def inspect_more():
    db_path = r"c:\Users\ROG\Documents\AIAPHotel\data\noshow.db"
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM noshow", conn)
    conn.close()
    
    # 1. Find the row(s) with missing target/values
    nan_target_rows = df[df['no_show'].isnull()]
    print("Row with missing no_show:")
    print(nan_target_rows)
    
    # 2. Check checkout_day distinct values
    print("\nCheckout day unique values (top 20):")
    print(df['checkout_day'].value_counts(dropna=False).head(20))
    
    # 3. Check arrival_month all unique values
    print("\nArrival month unique values:")
    print(df['arrival_month'].value_counts(dropna=False).head(30))
    
    # 4. Check price currencies and their patterns
    print("\nPrice prefixes:")
    price_non_null = df['price'].dropna()
    prefixes = price_non_null.apply(lambda x: str(x).split()[0] if len(str(x).split()) > 0 else 'EMPTY')
    print(prefixes.value_counts())
    
    # Let's check relation between price currency and country or branch
    df['currency'] = df['price'].apply(lambda x: str(x).split()[0] if pd.notnull(x) else None)
    print("\nCross-tab of Currency vs Country:")
    print(pd.crosstab(df['country'], df['currency'], margins=True))
    
    print("\nCross-tab of Currency vs Branch:")
    print(pd.crosstab(df['branch'], df['currency'], margins=True))

if __name__ == "__main__":
    inspect_more()
