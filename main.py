from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import base64
from io import BytesIO
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    except Exception as e:
        print("❌ CSV Read Error:", e)
        return JSONResponse(status_code=500, content={"error": f"Error reading CSV: {str(e)}"})

    try:
        df.columns = [col.strip().capitalize() for col in df.columns]
        df["Type"] = df["Type"].str.lower()
        df["Category"] = df["Category"].fillna("Uncategorized")

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

        def fig_to_base64():
            buf = BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")

        # Line Chart
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=monthly_expense.index.astype(str), y=monthly_expense.values, marker='o', label='Actual')
        sns.lineplot(x=[str(m.strftime('%b %Y')) for m in future_months], y=future_predictions, marker='o', linestyle='--', label='Predicted')
        plt.title("Monthly Expense Trend")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        line_chart = fig_to_base64()

        # Bar Chart
        plt.figure(figsize=(10, 5))
        sns.barplot(x=df["Category"].value_counts().index, y=df["Category"].value_counts().values)
        plt.title("Expense Categories Distribution")
        plt.xticks(rotation=45)
        plt.grid(True)
        bar_chart = fig_to_base64()

        # Pie Chart
        plt.figure(figsize=(7, 7))
        df.groupby("Category")["Amount"].sum().plot(kind='pie', autopct='%1.1f%%')
        plt.title("Spending by Category")
        plt.ylabel('')
        pie_chart = fig_to_base64()

           budget_suggestions = []
        for cat, spent in df.groupby("Category")["Amount"].sum().items():
        # Suggested budget is 90% of current spending for savings-oriented advice
        suggested = round(spent * 0.9, 2)
        budget_suggestions.append({
            "category": cat,
            "suggestedBudget": suggested,
            "currentSpending": round(spent, 2)
        })
    
        return JSONResponse(content={
        "insights": insights,
        "future_predictions": future_insights,
        "recommendations": recommendations,
        "budgetSuggestions": budget_suggestions,  # 🔥 Newly added
        "charts": {
            "line_chart_base64": line_chart,
            "bar_chart_base64": bar_chart,
            "pie_chart_base64": pie_chart
        }
    })

    except Exception as e:
        print("❌ Unexpected Error:", e)
        return JSONResponse(status_code=500, content={"error": f"Internal error: {str(e)}"})
