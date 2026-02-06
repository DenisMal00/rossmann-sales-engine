import pandas as pd
from sqlalchemy import create_engine
import os

DB_URL = "postgresql://admin:admin@db:5432/rossmann_db"


def main():
    engine = create_engine(DB_URL)
    print("Connected to database.")

    # 1. Handle store metadata
    print("Reading store.csv...")
    store_df = pd.read_csv('data/store.csv')

    # Matching column names to our SQL schema
    store_df.columns = [
        'store_id', 'store_type', 'assortment', 'competition_distance',
        'competition_open_since_month', 'competition_open_since_year',
        'promo2', 'promo2_since_week', 'promo2_since_year', 'promo_interval'
    ]

    print("Uploading store data...")
    store_df.to_sql('store', engine, if_exists='append', index=False)

    # 2. Handle historical sales data
    print("Reading train.csv... ")
    sales_df = pd.read_csv('data/train.csv', low_memory=False)

    # Map CSV headers to database columns
    sales_df.rename(columns={
        'Store': 'store_id', 'DayOfWeek': 'day_of_week', 'Date': 'date',
        'Sales': 'sales', 'Customers': 'customers', 'Open': 'open',
        'Promo': 'promo', 'StateHoliday': 'state_holiday', 'SchoolHoliday': 'school_holiday'
    }, inplace=True)

    print("Uploading sales in chunks of 50,000 rows...")
    sales_df.to_sql('sales', engine, if_exists='append', index=False, chunksize=50000)

    print("Ingestion finished successfully.")


if __name__ == "__main__":
    main()