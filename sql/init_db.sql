CREATE TABLE IF NOT EXISTS store (
    store_id INTEGER PRIMARY KEY,
    store_type VARCHAR(5),
    assortment VARCHAR(5),
    competition_distance FLOAT,
    competition_open_since_month FLOAT,
    competition_open_since_year FLOAT,
    promo2 INTEGER,
    promo2_since_week FLOAT,
    promo2_since_year FLOAT,
    promo_interval VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS sales (
    store_id INTEGER,
    day_of_week INTEGER,
    date DATE,
    sales INTEGER,
    customers INTEGER,
    open INTEGER,
    promo INTEGER,
    state_holiday VARCHAR(5),
    school_holiday INTEGER,
    PRIMARY KEY (store_id, date)
);

CREATE INDEX IF NOT EXISTS idx_date ON sales(date);
CREATE INDEX IF NOT EXISTS idx_store ON sales(store_id);