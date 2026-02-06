# Rossmann Strategic Dashboard
## Scalable Sales Forecasting & Strategy Simulator

> **Quick Start:** Skip to [Installation & Setup](#-installation--setup) at the bottom.

**Rossmann Strategic Dashboard** is a sales forecasting tool that transforms over **1.1 million historical records** into actionable business intelligence. By processing data from a massive network of **1,115 stores across Germany**, the system leverages a **Recursive LSTM** neural network to provide precise sales forecasts, achieving a solid **8.63% MAPE** on unseen data and simulate strategic scenarios in real-time.

Designed as a functional **Strategy Simulator**, this dashboard bridges the gap between historical time-series data and daily operational decisions. It allows users to visualize the potential impact of promotions and seasonal shifts, providing a data-driven baseline before implementing changes in the physical store.

---

## The Power of Data at Scale
The core of this project is built to handle significant data complexity:

* **Efficient Learning**: The model was trained on a high-granularity dataset, capturing essential sales patterns, holiday effects, and regional dynamics.
* **Store-Specific Logic**: The system accounts for the unique identity of every store, considering its competitive landscape and assortment strategy.
* **Practical Business Value**: In retail, even small forecasting errors can lead to inventory issues. This tool helps mitigate uncertainty by modeling recurring seasonal patterns and periodic market fluctuations.
---

## Dashboard Overview
The interface is structured into focused modules for immediate operational clarity:

### 🔍 Store Profiler & Metadata
The system performs real-time queries to identify the store's context.
* **Automatic Categorization**: Displays Store Type (Standard, Extra, Compact, Extended) and Assortment levels.
* **Competitive Landscape**: Shows the distance to the nearest competitor, a key factor used by the AI to weigh sales elasticity.
* **Helpful Context**: A specialized tooltip provides definitions of store archetypes for quick reference.

### 🎮 Strategy Simulator (What-If Analysis)
The heart of the dashboard is the ability to test scenarios through simulation.
* **Promo Toggle**: Inject promotional flags into the 7-day forecast window with a single click.
* **Dual Inference**: The engine generates two parallel scenarios: the **Baseline** (Business as Usual) and the **Strategic Scenario** (with active promotion).

### 📈 Predictive Visualization
An interactive **Altair-based** chart provides a clear view of time continuity:
* **History-to-Forecast Bridge**: A visual demarcation separates historical data from future AI projections.
* **Granular Inspection**: Hover states allow for detailed checks of forecasted values for every single day.

### 📊 Performance Analytics
Raw data is translated into simple, high-level KPIs:
* **Projected Performance (WoW)**: A direct indicator comparing the 7-day forecast against the **previous 7 days of actual history**.
* **Net Strategic Impact**: When a promo is active, the system isolates the estimated incremental revenue attributable to the strategy.

---

## 🛠 Engineering & Design Choices
*[Section in progress: Awaiting technical code for Train/Evaluate modules]*

---

## 📦 Installation & Setup

To run the entire ecosystem (PostgreSQL Database + Streamlit Dashboard) locally, ensure you have Docker installed.

### 1. Clone the repository
```bash
git clone [https://github.com/your-username/rossmann-forecast.git](https://github.com/your-username/rossmann-forecast.git)
cd rossmann-forecast
