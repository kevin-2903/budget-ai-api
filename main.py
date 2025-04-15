from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import shutil, os, base64
from io import BytesIO

app = FastAPI()

@app.post("/analyze/")
async def analyze_budget_data(file: UploadFile = File(...), income: float = Form(...)):
    # Save file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Read and process CSV
    df = pd.read_csv(temp_path, parse_dates=["date"])
    os.remove(temp_path)
    df["Month"] = df["date"].dt.to_period("M")

    monthly_expense = df.groupby('Month')['Amount'].sum()
    insights = "### Monthly Expense Analysis ###\n"
    for month, expense in monthly_expense.items():
        insights += f"{month}: ₹{expense:.2f}\n"
    insights += f"\nTotal Income: ₹{income}\n"
    insights += f"Average Monthly Expense: ₹{monthly_expense.mean():.2f}\n"
    insights += f"Estimated Savings: ₹{income - monthly_expense.mean():.2f}\n"

    # Prediction
    if len(monthly_expense) < 12:
        growth_rate = monthly_expense.pct_change().mean() if len(monthly_expense) > 1 else 0.05
        last_value = monthly_expense.iloc[-1]
        future_predictions = [last_value * (1 + growth_rate) ** i for i in range(1, 4)]
        future_months = [monthly_expense.index[-1] + i for i in range(1, 4)]
    else:
        model = ExponentialSmoothing(monthly_expense, trend='add', seasonal='add', seasonal_periods=12)
        model_fit = model.fit()
        future_months = [monthly_expense.index[-1] + i for i in range(1, 4)]
        future_predictions = model_fit.forecast(3)

    future_insights = "\n### Future Budget Predictions ###\n"
    for month, prediction in zip(future_months, future_predictions):
        future_insights += f"{month}: ₹{prediction:.2f}\n"

    # Recommendations
    category_expense = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
    top_categories = category_expense.head(3)
    recommendations = "\n### Savings Recommendations ###\n"
    recommendations += "1. Reduce spending in the following categories:\n"
    for cat, amt in top_categories.items():
        recommendations += f"   - {cat}: ₹{amt:.2f}\n"
    total_expense = df['Amount'].sum()
    savings_potential = income - total_expense
    if savings_potential > 0:
        recommendations += f"\n2. Surplus: ₹{savings_potential:.2f}. Consider investing.\n"
    else:
        recommendations += f"\n2. Deficit: ₹{-savings_potential:.2f}. Reduce discretionary expenses.\n"
    recommendations += "\n3. Suggested strategies:\n- Budget category limits\n- Use cash\n- Explore offers\n- Automate savings\n"

    # --- Graphs as base64 ---
    def fig_to_base64():
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    # Line chart: Monthly + Prediction
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=monthly_expense.index.astype(str), y=monthly_expense.values, marker='o', label='Actual')
    sns.lineplot(x=[str(m) for m in future_months], y=future_predictions, marker='o', linestyle='--', label='Predicted')
    plt.title("Monthly Expense Trend")
    line_chart = fig_to_base64()

    # Bar chart: Category Frequency
    plt.figure(figsize=(10, 5))
    sns.barplot(x=df['Category'].value_counts().index, y=df['Category'].value_counts().values)
    plt.title("Expense Categories Distribution")
    plt.xticks(rotation=45)
    bar_chart = fig_to_base64()

    # Pie chart: Spending by Category
    plt.figure(figsize=(7, 7))
    df.groupby('Category')['Amount'].sum().plot(kind='pie', autopct='%1.1f%%')
    plt.title("Spending by Category")
    plt.ylabel('')
    pie_chart = fig_to_base64()

    # Final response
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
