"""
This module performs all feature engineering calculations for the machine learning model.

I designed this module to engineer high-impact features from the raw date columns:
  - 'stay_duration': The duration of the reservation stay in days, calculated safely 
    across month boundaries by checking day limits in each month.
  - 'lead_time_months': The lead time of the booking (how early the guest booked), 
    calculated as the modular difference in months between the booking month and arrival month.
"""

import pandas as pd
from src.logger import get_logger

logger = get_logger("Features")

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes stay_duration and lead_time_months features from booking details.

    I have designed this function to construct high-impact predictive features:
      - 'stay_duration': Represents length of stay in days, correctly handling month 
        boundaries (e.g. arrival June 30, checkout July 2) via specific day caps.
      - 'lead_time_months': Represents how far in advance the customer made the
        booking, calculated as the modulo-12 month difference between arrival and booking.

    Args:
        df: A pd.DataFrame containing the cleaned hotel booking records.

    Returns:
        A pd.DataFrame with engineered 'stay_duration' and 'lead_time_months' features.

    Raises:
        KeyError: If required raw month/day columns are missing from the dataframe.
    """
    logger.info("Starting feature engineering calculations...")
    df = df.copy()
    
    # Check if necessary columns exist
    required_cols = ['arrival_month', 'checkout_month', 'booking_month', 'arrival_day', 'checkout_day']
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        err_msg = f"Missing required columns for feature engineering: {missing_cols}"
        logger.error(err_msg)
        raise KeyError(err_msg)
        
    # Map month names to numbers
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    
    df['arrival_month_num'] = df['arrival_month'].map(month_map)
    df['checkout_month_num'] = df['checkout_month'].map(month_map)
    df['booking_month_num'] = df['booking_month'].map(month_map)
    
    logger.info("Computing stay_duration feature...")
    def calculate_stay_duration(row):
        m1, d1 = row['arrival_month_num'], row['arrival_day']
        m2, d2 = row['checkout_month_num'], row['checkout_day']
        
        if pd.isnull(m1) or pd.isnull(m2) or pd.isnull(d1) or pd.isnull(d2):
            return 1
            
        days_in_month = {
            1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
            7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
        }
        
        if m1 == m2:
            return d2 - d1
        elif (m2 - m1) % 12 == 1:
            days_m1 = days_in_month.get(m1, 30)
            return (days_m1 - d1) + d2
        else:
            # Approximator for complex stays
            return (m2 - m1) % 12 * 30 + (d2 - d1)
            
    df['stay_duration'] = df.apply(calculate_stay_duration, axis=1)
    # Correct any zero or negative durations to 1
    df['stay_duration'] = df['stay_duration'].apply(lambda x: x if x > 0 else 1)
    
    logger.info("Computing lead_time_months feature...")
    df['lead_time_months'] = (df['arrival_month_num'] - df['booking_month_num']) % 12
    
    # Drop intermediate columns
    df = df.drop(columns=['arrival_month_num', 'checkout_month_num', 'booking_month_num'])
    
    logger.info(f"Feature engineering completed. Columns stay_duration and lead_time_months added.")
    return df
