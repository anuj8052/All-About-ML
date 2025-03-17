### **ARIMA Model: AutoRegressive Integrated Moving Average**  

The **ARIMA (AutoRegressive Integrated Moving Average)** model is one of the most widely used statistical methods for **time series forecasting**. It captures trends, seasonality, and patterns in data to make future predictions.

---

## **1️⃣ What is ARIMA?**
ARIMA is a combination of three key components:  
- **AR (AutoRegressive)** → Uses past values to predict future values.  
- **I (Integrated)** → Differencing is applied to make the series stationary.  
- **MA (Moving Average)** → Uses past errors to improve predictions.  

🔹 ARIMA is best for **univariate time series forecasting** (i.e., predicting based on past values of the same variable).  

---

## **2️⃣ Understanding ARIMA Components**
An ARIMA model is defined as **ARIMA(p, d, q)** where:

| Parameter | Meaning | Role |
|-----------|---------|------|
| **p** | AutoRegressive (AR) | Number of past observations to use. |
| **d** | Differencing (I) | Number of times to make data stationary. |
| **q** | Moving Average (MA) | Number of past error terms to use. |

### **Example: ARIMA(2,1,1)**
- **p = 2** → Uses the past 2 observations.  
- **d = 1** → Differenced once to make it stationary.  
- **q = 1** → Uses 1 past error term for smoothing.

---

## **3️⃣ When to Use ARIMA?**
✅ **Works well when:**
- The time series is **stationary** (no trend or seasonality).  
- You need **short-term** or **medium-term** forecasting.  
- The dataset is relatively **small to medium** (less than 100K observations).  

🚫 **Not ideal when:**
- Data has strong **seasonality** → Use **SARIMA** instead.  
- It’s a **multivariate problem** → Use **VAR models** or deep learning.  
- There are **external factors** influencing the series → Use **Exogenous Variables (ARIMAX)**.

---

## **4️⃣ How to Build an ARIMA Model?**
### **📌 Step 1: Check for Stationarity**
- Use **ADF Test (Augmented Dickey-Fuller Test)** to check if the time series is stationary.  
- If **not stationary**, apply **differencing** (i.e., subtracting consecutive values).

### **📌 Step 2: Determine p, d, q Values**
- **Autocorrelation Function (ACF)** → Helps find **q** (MA order).  
- **Partial Autocorrelation Function (PACF)** → Helps find **p** (AR order).  
- **Differencing (d)** → Use until the series becomes stationary.

### **📌 Step 3: Train ARIMA Model**
Use Python's `statsmodels` library:
```python
from statsmodels.tsa.arima.model import ARIMA

# Load dataset (example: stock prices)
data = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119]  # Example data

# Fit ARIMA model (p=2, d=1, q=1)
model = ARIMA(data, order=(2,1,1))
model_fit = model.fit()

# Print model summary
print(model_fit.summary())
```

### **📌 Step 4: Forecast Future Values**
```python
forecast = model_fit.forecast(steps=5)  # Predict next 5 values
print(forecast)
```

---

## **5️⃣ ARIMA vs SARIMA vs ARIMAX**
| Model | When to Use? | Example Use Case |
|-------|-------------|------------------|
| **ARIMA** | Stationary, non-seasonal data | Stock prices, temperature data |
| **SARIMA** | Seasonal data | Monthly sales, weather patterns |
| **ARIMAX** | ARIMA with external variables | Forecasting demand based on economic factors |

---

## **6️⃣ Advantages & Disadvantages**
### ✅ **Advantages**
✔ Works well with small datasets  
✔ Good for short-term forecasting  
✔ Simple & interpretable  

### ❌ **Disadvantages**
✖ Struggles with seasonality (use SARIMA instead)  
✖ Requires stationarity (needs differencing)  
✖ Less accurate than deep learning for complex data  

---

## **🚀 Summary**
- ARIMA is a powerful model for **time series forecasting**.  
- It consists of **AR (AutoRegressive), I (Integration), and MA (Moving Average)** components.  
- It works best for **stationary** and **univariate** time series.  
- Use **ACF, PACF, and differencing** to determine `p, d, q` values.  
- If seasonality exists, use **SARIMA** instead.
