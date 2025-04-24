from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import base64
import plotly.express as px
import plotly.io as pio
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from io import BytesIO

app = FastAPI()

# Allow all origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set default Plotly theme
pio.templates.default = "plotly_white"

def plotly_to_base64(fig):
    img_bytes = fig.to_image(format="png", width=1000, height=500)
    return base64.b64encode(img_bytes).decode("utf-8")


@app.get("/")
async def root():
    return {"message": "AI Budget Analysis API is up and running!"}


@app.post("/analyze")
async def analyze_budget_data(request: Request):
    data = await request.json()
    csv_url = data.get("csvUrl")

    if not csv_url:
        return JSONResponse(status_code=400, content={"error": "CSV URL is missing."})

    try:
        df = pd.read_csv(csv_url, parse_dates=["date"])
        df.columns = [col.strip().capitalize() for col in df.columns]
        df["Type"] = df["Type"].str.lower()
        df["Category"] = df["Category"].fillna("Uncategorized")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.dropna(subset=["Date"], inplace=True)

        income_total = df[df["Type"] == "income"]["Amount"].sum()
        df = df[df["Type"] == "expense"]
        df["Month"] = df["Date"].dt.to_period("M")
        monthly_expense = df.groupby("Month")["Amount"].sum()

        insights = "### Monthly Expense Analysis ###\n"
        for month, expense in monthly_expense.items():
            insights += f"{month}: ₹{expense:.2f}\n"
        insights += f"\nTotal Income: ₹{income_total:.2f}\n"
        insights += f"Average Monthly Expense: ₹{monthly_expense.mean():.2f}\n"
        insights += f"Estimated Savings: ₹{income_total - monthly_expense.mean():.2f}\n"

        # Forecast
        try:
            if len(monthly_expense) < 12:
                growth_rate = monthly_expense.pct_change().mean() if len(monthly_expense) > 1 else 0.05
                last_value = monthly_expense.iloc[-1]
                future_predictions = [last_value * (1 + growth_rate) ** i for i in range(1, 4)]
                future_months = [monthly_expense.index[-1] + i for i in range(1, 4)]
            else:
                monthly_expense.index = monthly_expense.index.to_timestamp()
                model = ExponentialSmoothing(monthly_expense, trend='add', seasonal='add', seasonal_periods=12)
                model_fit = model.fit()
                future_months = [monthly_expense.index[-1] + pd.DateOffset(months=i) for i in range(1, 4)]
                future_predictions = model_fit.forecast(3)
        except Exception as e:
            print("❌ Forecast Error:", e)
            future_months = [monthly_expense.index[-1] + i for i in range(1, 4)]
            future_predictions = [monthly_expense.mean()] * 3

        future_insights = "\n### Future Budget Predictions ###\n"
        for month, prediction in zip(future_months, future_predictions):
            future_insights += f"{month.strftime('%b %Y')}: ₹{prediction:.2f}\n"

        # Top Categories
        category_expense = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
        top_categories = category_expense.head(3)

        recommendations = "\n### Savings Recommendations ###\n"
        recommendations += "1. Reduce spending in the following categories:\n"
        for cat, amt in top_categories.items():
            recommendations += f"   - {cat}: ₹{amt:.2f}\n"
        total_expense = df["Amount"].sum()
        savings_potential = income_total - total_expense
        if savings_potential > 0:
            recommendations += f"\n2. Surplus: ₹{savings_potential:.2f}. Consider investing.\n"
        else:
            recommendations += f"\n2. Deficit: ₹{-savings_potential:.2f}. Reduce discretionary expenses.\n"
        recommendations += "\n3. Suggested strategies:\n- Budget category limits\n- Use cash\n- Explore offers\n- Automate savings\n"

        # --- Modern Charts (Plotly) ---

        # Line Chart (Monthly Trend)
        line_fig = px.line(
            x=monthly_expense.index.astype(str),
            y=monthly_expense.values,
            labels={"x": "Month", "y": "Expenses (₹)"},
            title="Monthly Expense Trend"
        )
        line_fig.add_scatter(
            x=[m.strftime('%b %Y') for m in future_months],
            y=future_predictions,
            mode='lines+markers',
            name='Predicted'
        )
        line_chart = plotly_to_base64(line_fig)

        # Bar Chart (Category Distribution)
        bar_data = df["Category"].value_counts().reset_index()
        bar_fig = px.bar(bar_data, x='index', y='Category', title="Expense Categories Distribution", labels={"index": "Category", "Category": "Count"})
        bar_chart = plotly_to_base64(bar_fig)

        # Pie Chart (Spending by Category)
        pie_data = df.groupby("Category")["Amount"].sum().reset_index()
        pie_fig = px.pie(pie_data, values="Amount", names="Category", title="Spending by Category", hole=0.4)
        pie_chart = plotly_to_base64(pie_fig)

        return JSONResponse(content={
            "insights": insights,
            "future_predictions": future_insights,
            "recommendations": recommendations,
            "charts": {
                "line_chart_base64": line_chart,
                "bar_chart_base64": bar_chart,
                "pie_chart_base64": pie_chart
            }
        })

    except Exception as e:
        print("❌ Unexpected Error:", e)
        return JSONResponse(status_code=500, content={"error": f"Internal error: {str(e)}"})
