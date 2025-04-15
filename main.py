from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

app = FastAPI()

def read_csv(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df['Month'] = df['date'].dt.to_period('M')
    return df

def analyze_budget(df, income):
    monthly_expense = df.groupby('Month')['Amount'].sum()
    insights = "### Monthly Expense Analysis ###\n"

    for month, expense in monthly_expense.items():
        insights += f"{month}: ₹{expense:.2f}\n"

    insights += f"\nTotal Income: ₹{income}\n"
    insights += f"Average Monthly Expense: ₹{monthly_expense.mean():.2f}\n"
    insights += f"Estimated Savings: ₹{income - monthly_expense.mean():.2f}\n"

    return insights, monthly_expense

def predict_future_budget(monthly_expense):
    future_months = []
    future_predictions = []

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

    return future_months, future_predictions, future_insights

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
        recommendations += f"\n2. You have a surplus of ₹{savings_potential:.2f}. Consider investing in mutual funds, fixed deposits, or emergency savings.\n"
    else:
        recommendations += f"\n2. Your expenses exceed your income by ₹{-savings_potential:.2f}. Consider revising discretionary spending.\n"

    recommendations += "\n3. Suggested strategies:\n"
    recommendations += "   - Set a monthly budget for dining out, entertainment, and shopping.\n"
    recommendations += "   - Use cash instead of credit cards to avoid unnecessary spending.\n"
    recommendations += "   - Explore discounts, cashback offers, and subscriptions with lower costs.\n"
    recommendations += "   - Plan large expenses ahead to avoid financial stress.\n"
    recommendations += "   - Automate savings by setting up a fixed transfer to your savings account.\n"

    return recommendations

def generate_graphs(monthly_expense, future_months, future_predictions, df):
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=monthly_expense.index.astype(str), y=monthly_expense.values, marker='o', label='Actual')
    sns.lineplot(x=[str(m) for m in future_months], y=future_predictions, marker='o', linestyle='dashed', label='Predicted')
    plt.xlabel('Month')
    plt.ylabel('Expense (₹)')
    plt.title('Monthly Expense Trends')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('budget_trend.png')

    plt.figure(figsize=(10, 5))
    sns.barplot(x=df['Category'].value_counts().index, y=df['Category'].value_counts().values)
    plt.xlabel('Category')
    plt.ylabel('Frequency')
    plt.title('Expense Categories Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('category_distribution.png')

    plt.figure(figsize=(8, 8))
    df.groupby('Category')['Amount'].sum().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Spending by Category')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('spending_pie_chart.png')

@app.post("/analyze/")
def analyze_income(income: float = Form(...)):
    try:
        file_path = 'sample_expenses.csv'  # Make sure this file is present in your deployment
        df = read_csv(file_path)
        insights, monthly_expense = analyze_budget(df, income)
        future_months, future_predictions, future_insights = predict_future_budget(monthly_expense)
        recommendations = savings_recommendations(df, income)
        generate_graphs(monthly_expense, future_months, future_predictions, df)

        with open('budget_insights.txt', 'w', encoding='utf-8') as f:
            f.write(insights + '\n' + future_insights + '\n' + recommendations)

        return JSONResponse({
            "status": "success",
            "message": "Analysis complete.",
            "files": [
                "budget_insights.txt",
                "budget_trend.png",
                "category_distribution.png",
                "spending_pie_chart.png"
            ]
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
