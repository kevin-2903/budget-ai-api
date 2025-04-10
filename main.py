from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import base64
import io
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    userId: str
    income: float
    csv_data: str  # base64 encoded CSV

def analyze_budget(df, income):
    df['Month'] = df['date'].dt.to_period('M')
    monthly_expense = df.groupby('Month')['Amount'].sum()
    insights = "### Monthly Expense Analysis ###\n"

    for month, expense in monthly_expense.items():
        insights += f"{month}: ₹{expense:.2f}\n"

    avg_expense = monthly_expense.mean()
    savings = income - avg_expense
    savings_rate = (savings / income) * 100 if income else 0

    insights += f"\nTotal Income: ₹{income}\n"
    insights += f"Average Monthly Expense: ₹{avg_expense:.2f}\n"
    insights += f"Estimated Savings: ₹{savings:.2f}\n"
    insights += f"Savings Rate: {savings_rate:.2f}%\n"

    return insights, monthly_expense

def predict_future_budget(monthly_expense):
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

    return future_insights, future_months, future_predictions

def savings_recommendations(df, income):
    category_expense = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
    top_categories = category_expense.head(3)
    recommendations = "\n### Savings Recommendations ###\n"

    recommendations += "1. Reduce spending in the following categories:\n"
    for cat, amt in top_categories.items():
        recommendations += f"   - {cat}: ₹{amt:.2f}\n"

    total_expense = df['Amount'].sum()
    savings_potential = income - total_expense

    if savings_potential > 0:
        recommendations += f"\n2. You have a surplus of ₹{savings_potential:.2f}. Consider investing or saving.\n"
    else:
        recommendations += f"\n2. Your expenses exceed your income by ₹{-savings_potential:.2f}. Consider reducing discretionary spending.\n"

    return recommendations

def generate_base64_graphs(monthly_expense, future_months, future_predictions, df, income):
    graphs = {}

    # Line Graph
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=monthly_expense.index.astype(str), y=monthly_expense.values, marker='o', label='Actual')
    sns.lineplot(x=[str(m) for m in future_months], y=future_predictions, marker='o', linestyle='dashed', label='Predicted')
    plt.xlabel('Month')
    plt.ylabel('Expense (₹)')
    plt.title('Monthly Expense Trends')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graphs['trend'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # Category Frequency
    plt.figure(figsize=(10, 5))
    sns.barplot(x=df['Category'].value_counts().index, y=df['Category'].value_counts().values)
    plt.xlabel('Category')
    plt.ylabel('Frequency')
    plt.title('Expense Categories Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graphs['category'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # Pie Chart
    plt.figure(figsize=(8, 8))
    df.groupby('Category')['Amount'].sum().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Spending by Category')
    plt.ylabel('')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graphs['pie'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # Spending Rate (% of Income) Chart
    plt.figure(figsize=(10, 5))
    spending_rate = (monthly_expense / income) * 100
    sns.lineplot(x=monthly_expense.index.astype(str), y=spending_rate, marker='o')
    plt.xlabel('Month')
    plt.ylabel('% of Income Spent')
    plt.title('Monthly Spending Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graphs['spending_rate'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return graphs
    
  
@app.get("/")
def root():
    return {"message": "API is running 🚀"}

@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    csv_bytes = base64.b64decode(request.csv_data)
    df = pd.read_csv(io.BytesIO(csv_bytes), parse_dates=['date'])

    insights, monthly_expense = analyze_budget(df, request.income)
    future_insights, future_months, future_predictions = predict_future_budget(monthly_expense)
    recommendations = savings_recommendations(df, request.income)
    graphs = generate_base64_graphs(monthly_expense, future_months, future_predictions, df, request.income)

    return {
        "insights": insights,
        "predictions": future_insights,
        "recommendations": recommendations,
        "graphs": graphs
    }
