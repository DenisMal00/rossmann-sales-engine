"""
Retrieves global sales data joined with store metadata for rich feature engineering.
"""
import pandas as pd
from sqlalchemy import create_engine
import os

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASSWORD", "admin")
DB_NAME = os.getenv("DB_NAME", "rossmann_db")

DB_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}"


def get_connection():
    return create_engine(DB_URL)


def get_training_data():
    engine = get_connection()

    # The query correctly maps columns to their respective tables (s = sales, st = store)
    query = """
            WITH raw_data AS (
                SELECT 
                    s.store_id,
                    s.date,
                    s.sales,
                    s.day_of_week,
                    EXTRACT(MONTH FROM s.date) as month,
                    s.promo,
                    st.promo2, -- Changed from s.promo2 to st.promo2
                    s.school_holiday,
                    s.state_holiday,
                    st.store_type,
                    st.assortment,
                    st.competition_distance,
                    -- Rolling average calculated on the full chronological timeline
                    AVG(s.sales) OVER (
                        PARTITION BY s.store_id 
                        ORDER BY s.date 
                        ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
                    ) as rolling_avg_7
                FROM sales s
                JOIN store st ON s.store_id = st.store_id
            )
            SELECT * FROM raw_data 
            WHERE sales > 0 
            ORDER BY store_id, date ASC;
            """

    df = pd.read_sql(query, engine)
    df['date'] = pd.to_datetime(df['date'])

    # Fill missing competition distance with a safe extreme value
    if not df['competition_distance'].empty:
        max_dist = df['competition_distance'].max()
        df['competition_distance'] = df['competition_distance'].fillna(max_dist * 2)

    # Remove the first 7 days of each store (where rolling_avg_7 is NaN)
    return df.dropna()