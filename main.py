# main.py

import base64
import io
import os
import random
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="NIDZO Financial Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sns.set(style="whitegrid")
plt.rcParams.update({
    "axes.facecolor": "#fefefe",
    "figure.facecolor": "#ffffff",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "font.family": "DejaVu Sans",
})

class AnalysisRequest(BaseModel):
    csvUrl: Optional[str] = None
    sampleData: Optional[bool] = False

class Insight(BaseModel):
    title: str
    description: str
    type: str

class Recommendation(BaseModel):
    title: str
    description: str
    priority: str

class Projection(BaseModel):
    month: str
    projected: float
    actual: Optional[float] = 0

class CategoryBreakdown(BaseModel):
    category: str
    amount: float
    percentage: float

class BudgetCategorySuggestion(BaseModel):
    category: str
    currentSpending: float
    suggestedBudget: float

class AnalysisResponse(BaseModel):
    insights: List[Insight]
    recommendations: List[Recommendation]
    charts: Dict[str, str]
    savingsProjection: List[Projection]
    categoryBreakdown: List[CategoryBreakdown]
    budgetSuggestions: List[BudgetCategorySuggestion]

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": traceback.format_exc()},
    )

def generate_sample_data() -> pd.DataFrame:
    np.random.seed(42)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    categories = ['Food', 'Transport', 'Entertainment', 'Housing', 'Utilities', 'Shopping', 'Healthcare', 'Other']
    types = ['expense'] * 100 + ['income'] * 10
    data = []
    for t in types:
        date = random.choice(dates)
        amount = random.uniform(500, 5000) if t == 'expense' else random.uniform(10000, 50000)
        category = random.choice(categories) if t == 'expense' else 'Salary'
        data.append({
            'id': f'{t[:3]}-{random.randint(1000,9999)}',
            'type': t,
            'date': date,
            'amount': round(amount, 2),
            'category': category,
            'description': f'{category} {t}'
        })
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    return df

def load_data_from_csv(csv_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_url)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception:
        return generate_sample_data()

def analyze_financial_data(df: pd.DataFrame) -> Dict:
    expenses = df[df['type'] == 'expense']
    income = df[df['type'] == 'income']
    total_expenses = expenses['amount'].sum()
    total_income = income['amount'].sum()
    net_cashflow = total_income - total_expenses
    savings_rate = (net_cashflow / total_income * 100) if total_income > 0 else 0

    insights = [
        {"title": "Total Income", "description": f"Your total income is ₹{total_income:,.2f}", "type": "positive"},
        {"title": "Total Expenses", "description": f"Your total expenses are ₹{total_expenses:,.2f}", "type": "warning"},
        {"title": "Net Cashflow", "description": f"Net cashflow is ₹{net_cashflow:,.2f}", "type": "neutral"},
        {"title": "Savings Rate", "description": f"Your savings rate is {savings_rate:.1f}%", "type": "positive" if savings_rate >= 20 else "warning"},
    ]

    recommendations = []
    if savings_rate < 20:
        recommendations.append({
            "title": "Increase Savings",
            "description": "Your savings rate is below the recommended 20%. Reduce expenses or increase income.",
            "priority": "high"
        })
    recommendations.append({
        "title": "Apply 50/30/20 Rule",
        "description": "Allocate your income wisely: 50% needs, 30% wants, 20% savings.",
        "priority": "medium"
    })

    monthly_expenses = expenses.groupby(expenses['date'].dt.to_period('M'))['amount'].sum()
    avg_monthly_expense = monthly_expenses.tail(3).mean()
    savings_projection = []
    for i in range(1, 4):
        next_month = (datetime.now() + timedelta(days=30 * i)).strftime('%B %Y')
        predicted = max(0, random.uniform(0.8, 1.2) * (total_income / 12 - avg_monthly_expense))
        savings_projection.append({"month": next_month, "projected": round(predicted, 2), "actual": 0})

    pie_data = expenses.groupby('category')['amount'].sum()
    category_breakdown = [
        {"category": cat, "amount": amt, "percentage": (amt / total_expenses) * 100}
        for cat, amt in pie_data.items()
    ]

    budget_suggestions = [
        {
            "category": cat,
            "currentSpending": amt,
            "suggestedBudget": round(amt * random.uniform(0.8, 1.1), 2)
        }
        for cat, amt in pie_data.items()
    ]

    return {
        "insights": insights,
        "recommendations": recommendations,
        "savingsProjection": savings_projection,
        "expenses": expenses,
        "income": income,
        "categoryBreakdown": category_breakdown,
        "budgetSuggestions": budget_suggestions
    }

def create_charts(expenses: pd.DataFrame, income: pd.DataFrame) -> Dict[str, str]:
    charts = {}
    try:
        pie_data = expenses.groupby('category')['amount'].sum()
        colors = sns.color_palette("Set2", len(pie_data))

        # Pie Chart
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        wedges, texts, autotexts = ax1.pie(
            pie_data,
            labels=pie_data.index,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors,
            wedgeprops=dict(edgecolor='white')
        )
        for text in texts + autotexts:
            text.set_fontsize(10)
        ax1.set_title("Expense Distribution", fontsize=14)
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        charts["pieChart"] = base64.b64encode(buffer.getvalue()).decode()

        # Line Chart
        expenses['month'] = expenses['date'].dt.to_period('M')
        income['month'] = income['date'].dt.to_period('M')
        exp_monthly = expenses.groupby('month')['amount'].sum()
        inc_monthly = income.groupby('month')['amount'].sum()

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(exp_monthly.index.astype(str), exp_monthly.values, label="Expenses", marker="o", color="red")
        ax2.plot(inc_monthly.index.astype(str), inc_monthly.values, label="Income", marker="s", color="green")
        ax2.legend()
        ax2.set_title("Monthly Income vs Expenses")
        ax2.set_ylabel("Amount (₹)")
        plt.xticks(rotation=45)
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        charts["lineChart"] = base64.b64encode(buffer.getvalue()).decode()

        # Bar Chart
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        top_categories = pie_data.sort_values(ascending=False).head(5)
        sns.barplot(x=top_categories.values, y=top_categories.index, palette="coolwarm", ax=ax3)
        ax3.set_title("Top Expense Categories")
        ax3.set_xlabel("Amount (₹)")
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        charts["barChart"] = base64.b64encode(buffer.getvalue()).decode()

    except Exception as e:
        print(f"Chart generation failed: {e}")

    return charts

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    df = generate_sample_data() if not request.csvUrl else load_data_from_csv(request.csvUrl)
    results = analyze_financial_data(df)
    charts = create_charts(results["expenses"], results["income"])

    return {
        "insights": results["insights"],
        "recommendations": results["recommendations"],
        "charts": charts,
        "savingsProjection": results["savingsProjection"],
        "categoryBreakdown": results["categoryBreakdown"],
        "budgetSuggestions": results["budgetSuggestions"],
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
