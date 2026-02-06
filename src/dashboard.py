"""
Rossmann Strategic Dashboard.

This dashboard provides sales forecasting using a recursive LSTM model.
It features a comparison between a baseline and a strategic scenario,
including Week-over-Week (WoW) performance analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import altair as alt
import textwrap
from tensorflow.keras.models import load_model
from datetime import timedelta
from database import get_store_chart_data, get_store_model_context

# --- Configuration ---
st.set_page_config(page_title="Rossmann Strategic Dashboard", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sales_model.keras')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.joblib')

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    return load_model(MODEL_PATH, compile=False), joblib.load(SCALER_PATH)

def run_forecast(store_id, model, scaler, n_days, promo_active):
    # Fetch rich context data for the sliding window
    df = get_store_model_context(store_id, limit=50)
    if len(df) < 14: return pd.DataFrame()

    df['date'] = pd.to_datetime(df['date'])
    df['log_sales'] = np.log1p(df['sales'])
    df['month'] = df['date'].dt.month
    df['rolling_avg_7'] = df['sales'].rolling(7).mean().fillna(df['sales'].mean())

    # Features mapping
    df['store_type'] = df['store_type'].map({'a': 0, 'b': 1, 'c': 2, 'd': 3}).fillna(0)
    df['assortment'] = df['assortment'].map({'a': 0, 'b': 1, 'c': 2}).fillna(0)
    df['state_holiday'] = df['state_holiday'].astype(str).map({'0': 0, 'a': 1, 'b': 2, 'c': 3}).fillna(0)

    cols = ['log_sales', 'promo', 'promo2', 'school_holiday', 'state_holiday',
            'day_of_week', 'month', 'rolling_avg_7', 'competition_distance', 'store_type', 'assortment']

    last_date = df.iloc[-1]['date']
    # The bridge point helps connecting historical data with the forecast in the chart
    results = [{'date': last_date, 'sales': df.iloc[-1]['sales'], 'status': 'Forecast', 'is_bridge': True}]
    curr_df = df.copy()

    for i in range(n_days):
        target_date = last_date + timedelta(days=i + 1)

        # Scenario logic: Promo only during weekdays
        is_weekday = target_date.weekday() < 5
        p_flag = 1 if (promo_active and is_weekday) else 0

        window = curr_df[cols].tail(7)
        scaled = scaler.transform(window)

        # Recursive prediction
        pred_log = model.predict([scaled.reshape((1, 7, 11)), np.array([store_id])], verbose=0)[0][0]

        # Inverse transform to get actual sales
        dummy = np.zeros((1, 11))
        dummy[0, 0] = pred_log
        sales = np.expm1(scaler.inverse_transform(dummy)[0, 0])

        # Force zero sales on Sundays
        if target_date.weekday() == 6: sales = 0.0

        new_row = curr_df.iloc[-1].copy()
        new_row.update({
            'date': target_date, 'sales': sales, 'log_sales': np.log1p(sales),
            'promo': p_flag, 'school_holiday': 0,
            'day_of_week': target_date.weekday() + 1, 'month': target_date.month,
            'rolling_avg_7': (curr_df['sales'].tail(6).sum() + sales) / 7
        })

        curr_df = pd.concat([curr_df, pd.DataFrame([new_row])], ignore_index=True)
        results.append({'date': target_date, 'sales': sales, 'status': 'Forecast', 'is_bridge': False})

    return pd.DataFrame(results)

# --- State Management ---
if 'f_data' not in st.session_state: st.session_state.f_data = pd.DataFrame()
if 'b_data' not in st.session_state: st.session_state.b_data = pd.DataFrame()
if 'run_with_promo' not in st.session_state: st.session_state.run_with_promo = False

model, scaler = load_assets()

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")

    def reset_simulation():
        st.session_state.f_data = pd.DataFrame()
        st.session_state.b_data = pd.DataFrame()
        st.session_state.run_with_promo = False

    store_id = st.number_input("Store ID", 1, 1115, 100, on_change=reset_simulation)
    history_days = st.slider("History Window", 7, 90, 28)

    # Tooltip with Markdown legend
    legend_text = textwrap.dedent("""
        **Store Types**
        * Standard: Typical neighborhood store.
        * Extra: Large stores with high product variety.
        * Compact: Small footprint, high-density areas.
        * Extended: Large variety with specialized departments.

        **Assortment Levels**
        * Basic: Only essential items.
        * Extra: Medium product range.
        * Extended: Full premium product selection.

        **Competition Distance**
        Distance in meters to the nearest competitor store.
    """).strip()

    st.divider()
    st.subheader("Store Profile", help=legend_text)

    store_info = get_store_model_context(store_id, limit=1)
    if not store_info.empty:
        row = store_info.iloc[0]
        type_mapping = {'a': 'Standard', 'b': 'Extra', 'c': 'Compact', 'd': 'Extended'}
        assort_mapping = {'a': 'Basic', 'b': 'Extra', 'c': 'Extended'}

        st.write(f"**Type:** {type_mapping.get(row['store_type'].lower(), row['store_type'])}")
        st.write(f"**Assortment:** {assort_mapping.get(row['assortment'].lower(), row['assortment'])}")
        st.write(f"**Competition:** {row['competition_distance']:,}m")

    st.divider()
    st.header("Promo")
    promo_active = st.toggle("Activate Promo Week", value=False)

# --- Main UI ---
st.title("Rossmann Strategic Dashboard")

if model and scaler:
    # Fetch historical data for charting
    hist_raw = get_store_chart_data(store_id, history_days)
    hist_raw['date'] = pd.to_datetime(hist_raw['date'])
    hist_raw['status'] = 'History'
    hist_df = hist_raw.sort_values('date')

    if st.button("Run Simulation", use_container_width=True):
        with st.spinner("Analyzing strategy..."):
            st.session_state.f_data = run_forecast(store_id, model, scaler, 7, promo_active)
            st.session_state.b_data = run_forecast(store_id, model, scaler, 7, False)
            st.session_state.run_with_promo = promo_active

    f_df = st.session_state.f_data
    full_df = pd.concat([hist_df, f_df], ignore_index=True) if not f_df.empty else hist_df

    # --- Chart Rendering ---
    base = alt.Chart(full_df).encode(
        x=alt.X('date:T', axis=alt.Axis(format='%d %b', title=None, tickCount=10, labelOverlap=True))
    )

    history_area = base.transform_filter(alt.datum.status == 'History').mark_area(
        line={'color': '#3498db'}, color='#3498db', opacity=0.4
    ).encode(y='sales:Q')

    forecast_area = base.transform_filter(alt.datum.status == 'Forecast').mark_area(
        color='#e74c3c', opacity=0.25
    ).encode(y='sales:Q')

    forecast_line = base.transform_filter(alt.datum.status == 'Forecast').mark_line(
        color='#e74c3c', size=3
    ).encode(y='sales:Q')

    points = base.transform_filter((alt.datum.status == 'Forecast') & (alt.datum.is_bridge == False)).mark_point(
        color='#e74c3c', filled=True, size=50
    ).encode(y='sales:Q')

    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['date'], empty=False)
    selectors = base.mark_point().encode(opacity=alt.value(0)).add_params(nearest)
    rule = base.mark_rule(color='gray').encode(opacity=alt.condition(nearest, alt.value(0.5), alt.value(0))).transform_filter(nearest)

    tooltip_label = base.mark_text(align='left', dx=5, dy=-20).encode(
        text=alt.condition(nearest, alt.Text('sales:Q', format='€,.0f'), alt.value(' ')),
        y='sales:Q'
    ).transform_filter(nearest)

    st.altair_chart((history_area + forecast_area + forecast_line + points + selectors + rule + tooltip_label).properties(height=380).interactive(), use_container_width=True)

    # --- Results Analysis ---
    if not f_df.empty:
        st.divider()

        # Prepare actual vs predicted sets
        real_f = f_df[f_df['is_bridge'] == False].reset_index(drop=True)
        real_b = st.session_state.b_data[st.session_state.b_data['is_bridge'] == False].reset_index(drop=True)

        # 1. Projected Performance (WoW Comparison)
        st.subheader("Projected Performance")

        last_week_actual = hist_df['sales'].tail(7).sum()
        forecast_total = real_f['sales'].sum()
        wow_delta_pct = ((forecast_total - last_week_actual) / last_week_actual) * 100 if last_week_actual > 0 else 0

        st.metric(
            label="Total Forecasted Revenue (Next 7 Days)",
            value=f"€ {forecast_total:,.2f}",
            delta=f"{wow_delta_pct:+.1f}% vs Last Week Historical",
            delta_color="normal"
        )

        # 2. Strategic Impact Analysis (Visible only if a Strategy was simulated)
        if st.session_state.run_with_promo:
            st.divider()
            st.subheader("Strategic Impact Analysis")

            total_promo = real_f['sales'].sum()
            total_base = real_b['sales'].sum()
            net_impact = total_promo - total_base

            c1, c2, c3 = st.columns(3)
            c1.metric("Total with Promo", f"€ {total_promo:,.2f}")
            c2.metric("Total without Promo", f"€ {total_base:,.2f}")
            c3.metric("Net Promo Impact", f"€ {net_impact:,.2f}", delta=f"€ {net_impact:,.2f}")

            # Impact Table
            impact_df = pd.DataFrame({
                'Date': real_f['date'],
                'With Promo (€)': real_f['sales'],
                'Without Promo (€)': real_b['sales'],
                'Delta (€)': real_f['sales'] - real_b['sales']
            })

            def style_delta(v):
                if v > 0.01: return 'color: #27ae60; font-weight: bold'
                if v < -0.01: return 'color: #e74c3c; font-weight: bold'
                return ''

            st.dataframe(
                impact_df.style.format({
                    'Date': lambda t: t.strftime('%A, %d %b'),
                    'With Promo (€)': '€ {:,.2f}',
                    'Without Promo (€)': '€ {:,.2f}',
                    'Delta (€)': '€ {:+,.2f}'
                }).map(style_delta, subset=['Delta (€)']),
                use_container_width=True,
                hide_index=True
            )
        else:
            # Standard forecast details list
            st.write("### Forecast Details")
            simple_df = pd.DataFrame({
                'Date': real_f['date'],
                'Forecasted Sales (€)': real_f['sales']
            })

            st.dataframe(
                simple_df.style.format({
                    'Date': lambda t: t.strftime('%A, %d %b'),
                    'Forecasted Sales (€)': '€ {:,.2f}'
                }),
                use_container_width=True,
                hide_index=True
            )