import base64
import io
import json
import logging
import os
import random
import sys
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from urllib.request import urlopen
from io import StringIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("financial-ai")

# Initialize FastAPI app
app = FastAPI(
    title="Financial Analysis AI",
    description="AI model for analyzing financial data and generating insights",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up visualization style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['figure.facecolor'] = '#ffffff'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

# Color palettes
CATEGORY_COLORS = {
    "Food": "#FF6B6B",
    "Groceries": "#4ECDC4",
    "Dining": "#FF6B6B",
    "Transport": "#FFD166",
    "Entertainment": "#6A0572",
    "Shopping": "#F72585",
    "Housing": "#4361EE",
    "Utilities": "#3A86FF",
    "Healthcare": "#06D6A0",
    "Education": "#118AB2",
    "Travel": "#073B4C",
    "Subscriptions": "#7209B7",
    "Salary": "#38b000",
    "Investment": "#588157",
    "Other": "#adb5bd"
}

# Default color palettes
CHART_COLORS = {
    "primary": ["#4361EE", "#3A86FF", "#4CC9F0", "#4ECDC4", "#06D6A0"],
    "accent": ["#F72585", "#7209B7", "#3A0CA3", "#4361EE", "#4CC9F0"],
    "pastel": ["#FFD166", "#06D6A0", "#4ECDC4", "#118AB2", "#073B4C"],
    "categorical": sns.color_palette("tab10").as_hex()
}

# Define request and response models
class AnalysisRequest(BaseModel):
    csvUrl: Optional[str] = Field(None, description="URL to the CSV file")
    sampleData: Optional[bool] = Field(False, description="Use sample data if true")
    timeRange: Optional[str] = Field("all", description="Time range for analysis (all, month, quarter, year)")
    categories: Optional[List[str]] = Field(None, description="Categories to include in analysis")

class Insight(BaseModel):
    title: str = Field(..., description="Title of the insight")
    description: str = Field(..., description="Description of the insight")
    type: str = Field(..., description="Type of insight (positive, warning, neutral, info)")

class Recommendation(BaseModel):
    title: str = Field(..., description="Title of the recommendation")
    description: str = Field(..., description="Description of the recommendation")
    priority: str = Field(..., description="Priority of recommendation (high, medium, low)")
    action: Optional[str] = Field(None, description="Suggested action to take")

class BudgetSuggestion(BaseModel):
    category: str = Field(..., description="Expense category")
    currentSpending: float = Field(..., description="Current spending amount")
    suggestedBudget: float = Field(..., description="Suggested budget amount")
    percentChange: float = Field(..., description="Percent change from current to suggested")

class SpendingTrend(BaseModel):
    month: str = Field(..., description="Month name")
    amount: float = Field(..., description="Spending amount")
    category: str = Field(..., description="Category name (or 'All' for total)")

class SavingsProjection(BaseModel):
    month: str = Field(..., description="Month name")
    projected: float = Field(..., description="Projected savings amount")
    actual: Optional[float] = Field(0, description="Actual savings amount (if available)")

class CategoryBreakdown(BaseModel):
    category: str = Field(..., description="Category name")
    percentage: float = Field(..., description="Percentage of total expenses")
    amount: float = Field(..., description="Amount spent")

class Charts(BaseModel):
    pieChart: str = Field(..., description="Base64 encoded pie chart image")
    lineChart: str = Field(..., description="Base64 encoded line chart image")
    barChart: str = Field(..., description="Base64 encoded bar chart image")

class AnalysisResponse(BaseModel):
    insights: List[Insight] = Field(..., description="List of financial insights")
    recommendations: List[Recommendation] = Field(..., description="List of recommendations")
    budgetSuggestions: List[BudgetSuggestion] = Field(..., description="List of budget suggestions")
    spendingTrends: List[SpendingTrend] = Field(..., description="Monthly spending trends")
    savingsProjection: List[SavingsProjection] = Field(..., description="Savings projections")
    categoryBreakdown: List[CategoryBreakdown] = Field(..., description="Expense category breakdown")
    charts: Charts = Field(..., description="Generated charts")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred",
            "message": str(exc),
            "traceback": traceback.format_exc() if os.getenv("DEBUG") == "true" else None
        },
    )

def generate_sample_data() -> pd.DataFrame:
    """Generate sample financial data for testing"""
    logger.info("Generating sample financial data")
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Define date range (last 12 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Define categories and transaction types
    expense_categories = [
        "Food", "Groceries", "Transport", "Entertainment", "Shopping", 
        "Housing", "Utilities", "Healthcare", "Education", "Travel", 
        "Subscriptions", "Other"
    ]
    income_categories = ["Salary", "Investment", "Other"]
    
    # Generate transactions
    data = []
    
    # Generate regular income (monthly salary)
    for month in pd.date_range(start=start_date, end=end_date, freq='M'):
        # Monthly salary (around 15th of month)
        salary_date = month.replace(day=min(15, month.days_in_month))
        salary_amount = np.random.normal(50000, 5000)  # Mean 50k with some variation
        data.append({
            'id': f'inc-{len(data):04d}',
            'date': salary_date,
            'description': 'Monthly Salary',
            'category': 'Salary',
            'type': 'income',
            'amount': max(round(salary_amount, 2), 0)
        })
        
        # Occasional investment income
        if np.random.random() < 0.3:  # 30% chance each month
            inv_date = month.replace(day=np.random.randint(1, month.days_in_month))
            inv_amount = np.random.uniform(1000, 10000)
            data.append({
                'id': f'inc-{len(data):04d}',
                'date': inv_date,
                'description': 'Investment Return',
                'category': 'Investment',
                'type': 'income',
                'amount': round(inv_amount, 2)
            })
    
    # Generate expenses
    for _ in range(500):  # Generate 500 expense transactions
        date = random.choice(dates)
        category = random.choice(expense_categories)
        
        # Amount depends on category
        if category == "Housing":
            amount = np.random.normal(15000, 2000)
        elif category == "Utilities":
            amount = np.random.normal(3000, 500)
        elif category in ["Food", "Groceries"]:
            amount = np.random.normal(2000, 800)
        elif category == "Transport":
            amount = np.random.normal(1500, 500)
        else:
            amount = np.random.exponential(1000)
            
        # Description based on category
        if category == "Food":
            description = random.choice(["Restaurant", "Cafe", "Food Delivery", "Lunch"])
        elif category == "Groceries":
            description = random.choice(["Supermarket", "Grocery Store", "Market"])
        elif category == "Transport":
            description = random.choice(["Fuel", "Taxi", "Public Transport", "Car Service"])
        elif category == "Entertainment":
            description = random.choice(["Movies", "Concert", "Games", "Streaming"])
        elif category == "Shopping":
            description = random.choice(["Clothing", "Electronics", "Home Goods", "Online Shopping"])
        else:
            description = f"{category} Expense"
            
        data.append({
            'id': f'exp-{len(data):04d}',
            'date': date,
            'description': description,
            'category': category,
            'type': 'expense',
            'amount': max(round(amount, 2), 0)
        })
    
    # Convert to DataFrame and sort by date
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    logger.info(f"Generated {len(df)} sample transactions")
    return df

async def load_data_from_csv(csv_url: str) -> pd.DataFrame:
    """Load financial data from a CSV URL"""
    try:
        logger.info(f"Loading data from CSV URL: {csv_url}")
        
        # Handle different URL formats
        if csv_url.startswith(('http://', 'https://')):
            response = urlopen(csv_url)
            csv_data = response.read().decode('utf-8')
            df = pd.read_csv(StringIO(csv_data))
        else:
            # Assume it's a local file path
            df = pd.read_csv(csv_url)
        
        # Ensure required columns exist
        required_columns = ['date', 'amount', 'type', 'category']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns in CSV: {missing_columns}")
            raise ValueError(f"CSV is missing required columns: {', '.join(missing_columns)}")
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Ensure amount is numeric
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Drop rows with NaN values
        df = df.dropna(subset=['date', 'amount', 'type', 'category'])
        
        logger.info(f"Successfully loaded {len(df)} transactions from CSV")
        return df
        
    except Exception as e:
        logger.error(f"Error loading CSV data: {str(e)}")
        logger.error(traceback.format_exc())
        logger.info("Falling back to sample data")
        return generate_sample_data()

def filter_data_by_time_range(df: pd.DataFrame, time_range: str) -> pd.DataFrame:
    """Filter data based on specified time range"""
    today = datetime.now()
    
    if time_range == "month":
        start_date = datetime(today.year, today.month, 1)
        return df[df['date'] >= start_date]
    
    elif time_range == "quarter":
        current_quarter = (today.month - 1) // 3 + 1
        start_month = (current_quarter - 1) * 3 + 1
        start_date = datetime(today.year, start_month, 1)
        return df[df['date'] >= start_date]
    
    elif time_range == "year":
        start_date = datetime(today.year, 1, 1)
        return df[df['date'] >= start_date]
    
    elif time_range == "all":
        return df
    
    else:
        logger.warning(f"Unknown time range: {time_range}, using all data")
        return df

def analyze_financial_data(df: pd.DataFrame) -> Dict:
    """Analyze financial data and generate insights"""
    logger.info("Analyzing financial data")
    
    # Split data into expenses and income
    expenses = df[df['type'] == 'expense'].copy()
    income = df[df['type'] == 'income'].copy()
    
    # Calculate basic metrics
    total_expenses = expenses['amount'].sum()
    total_income = income['amount'].sum()
    net_cashflow = total_income - total_expenses
    savings_rate = (net_cashflow / total_income * 100) if total_income > 0 else 0
    
    # Monthly analysis
    expenses['month'] = expenses['date'].dt.to_period('M')
    income['month'] = income['date'].dt.to_period('M')
    
    monthly_expenses = expenses.groupby('month')['amount'].sum()
    monthly_income = income.groupby('month')['amount'].sum()
    
    # Category analysis
    category_expenses = expenses.groupby('category')['amount'].sum().sort_values(ascending=False)
    category_percentages = (category_expenses / total_expenses * 100).round(1)
    
    # Top spending categories
    top_categories = category_expenses.head(5)
    
    # Monthly trends
    expense_trend = monthly_expenses.tail(6)
    income_trend = monthly_income.tail(6)
    
    # Calculate average monthly expense
    avg_monthly_expense = monthly_expenses.mean()
    
    # Generate insights
    insights = [
        {
            "title": "Total Income",
            "description": f"Your total income is ₹{total_income:,.2f}",
            "type": "positive" if total_income > 0 else "warning"
        },
        {
            "title": "Total Expenses",
            "description": f"Your total expenses are ₹{total_expenses:,.2f}",
            "type": "warning" if total_expenses > 0.8 * total_income else "neutral"
        },
        {
            "title": "Net Cashflow",
            "description": f"Your net cashflow is ₹{net_cashflow:,.2f}",
            "type": "positive" if net_cashflow > 0 else "warning"
        },
        {
            "title": "Savings Rate",
            "description": f"Your savings rate is {savings_rate:.1f}%",
            "type": "positive" if savings_rate >= 20 else "warning"
        }
    ]
    
    # Add category insights
    for category, amount in top_categories.items():
        percentage = category_percentages[category]
        insights.append({
            "title": f"Top Spending: {category}",
            "description": f"You spent ₹{amount:,.2f} ({percentage:.1f}%) on {category}",
            "type": "info"
        })
    
    # Monthly trend insight
    if len(expense_trend) >= 2:
        last_month = expense_trend.index[-1]
        prev_month = expense_trend.index[-2]
        month_change = ((expense_trend.iloc[-1] - expense_trend.iloc[-2]) / expense_trend.iloc[-2] * 100)
        
        insights.append({
            "title": "Monthly Trend",
            "description": f"Your spending in {last_month} {'increased' if month_change > 0 else 'decreased'} by {abs(month_change):.1f}% compared to {prev_month}",
            "type": "warning" if month_change > 10 else "positive" if month_change < 0 else "info"
        })
    
    # Generate recommendations
    recommendations = []
    
    # Savings rate recommendation
    if savings_rate < 20:
        recommendations.append({
            "title": "Increase Savings Rate",
            "description": "Your savings rate is below the recommended 20%. Consider reducing discretionary expenses.",
            "priority": "high",
            "action": "Aim to save at least 20% of your income"
        })
    
    # Category-specific recommendations
    for category, amount in top_categories.items():
        percentage = category_percentages[category]
        
        # Thresholds for different categories
        if category == "Food" and percentage > 15:
            recommendations.append({
                "title": "Reduce Food Expenses",
                "description": f"Your spending on Food ({percentage:.1f}%) is higher than recommended. Consider cooking at home more often.",
                "priority": "medium",
                "action": "Limit eating out to once a week"
            })
        elif category == "Entertainment" and percentage > 10:
            recommendations.append({
                "title": "Reduce Entertainment Expenses",
                "description": f"Your Entertainment expenses ({percentage:.1f}%) are higher than recommended.",
                "priority": "medium",
                "action": "Look for free or low-cost entertainment options"
            })
        elif category == "Shopping" and percentage > 15:
            recommendations.append({
                "title": "Reduce Shopping Expenses",
                "description": f"Your Shopping expenses ({percentage:.1f}%) are higher than recommended.",
                "priority": "medium",
                "action": "Implement a 24-hour rule before making non-essential purchases"
            })
    
    # Budget allocation recommendation
    recommendations.append({
        "title": "Follow 50/30/20 Rule",
        "description": "Allocate 50% of income to needs, 30% to wants, and 20% to savings/debt repayment.",
        "priority": "medium",
        "action": "Review your budget categories and reallocate as needed"
    })
    
    # Emergency fund recommendation
    recommendations.append({
        "title": "Build Emergency Fund",
        "description": "Aim to have 3-6 months of expenses saved in an emergency fund.",
        "priority": "high",
        "action": "Set up automatic transfers to a separate emergency fund account"
    })
    
    # Create budget suggestions
    budget_suggestions = []
    for category, amount in category_expenses.items():
        # Skip categories with very small amounts
        if amount < 0.01 * total_expenses:
            continue
            
        # Calculate suggested budget based on category
        if category == "Housing":
            # Housing should be around 30% of income
            suggested_percentage = 30
            suggested_budget = total_income * 0.3
        elif category == "Food" or category == "Groceries":
            # Food should be around 10-15% of income
            suggested_percentage = 15
            suggested_budget = total_income * 0.15
        elif category == "Transport":
            # Transport should be around 10% of income
            suggested_percentage = 10
            suggested_budget = total_income * 0.1
        elif category == "Entertainment" or category == "Shopping":
            # Discretionary spending should be reduced if savings rate is low
            if savings_rate < 20:
                suggested_budget = amount * 0.8  # Suggest 20% reduction
            else:
                suggested_budget = amount  # Keep current spending
        else:
            # For other categories, suggest slight reduction if spending is high
            if amount > 0.1 * total_income:
                suggested_budget = amount * 0.9  # Suggest 10% reduction
            else:
                suggested_budget = amount  # Keep current spending
                
        # Calculate percent change
        percent_change = ((suggested_budget - amount) / amount * 100) if amount > 0 else 0
        
        budget_suggestions.append({
            "category": category,
            "currentSpending": amount,
            "suggestedBudget": suggested_budget,
            "percentChange": percent_change
        })
    
    # Create spending trends
    spending_trends = []
    for month, amount in monthly_expenses.tail(6).items():
        spending_trends.append({
            "month": month.strftime("%b"),
            "amount": amount,
            "category": "All"
        })
    
    # Create category breakdown
    category_breakdown = []
    for category, amount in category_expenses.items():
        percentage = (amount / total_expenses * 100) if total_expenses > 0 else 0
        category_breakdown.append({
            "category": category,
            "amount": amount,
            "percentage": percentage
        })
    
    # Create savings projections
    savings_projection = []
    last_month_expense = monthly_expenses.iloc[-1] if len(monthly_expenses) > 0 else avg_monthly_expense
    last_month_income = monthly_income.iloc[-1] if len(monthly_income) > 0 else total_income / len(monthly_income) if len(monthly_income) > 0 else 0
    
    # Project for next 6 months
    today = datetime.now()
    for i in range(1, 7):
        projection_date = today + timedelta(days=30*i)
        month_name = projection_date.strftime("%b %Y")
        
        # Project income (slight increase)
        projected_income = last_month_income * (1 + 0.01 * i)  # 1% increase per month
        
        # Project expenses (based on historical data with some randomness)
        projected_expense = last_month_expense * (1 + np.random.normal(0, 0.05))  # +/- 5% variation
        
        # Calculate projected savings
        projected_savings = projected_income - projected_expense
        
        # For past months, we might have actual data
        actual_savings = 0
        if i <= 3 and len(monthly_income) > i and len(monthly_expenses) > i:
            actual_income = monthly_income.iloc[-(i+1)]
            actual_expense = monthly_expenses.iloc[-(i+1)]
            actual_savings = actual_income - actual_expense
        
        savings_projection.append({
            "month": month_name,
            "projected": projected_savings,
            "actual": actual_savings if i <= 3 else 0
        })
    
    return {
        "insights": insights,
        "recommendations": recommendations,
        "budgetSuggestions": budget_suggestions,
        "spendingTrends": spending_trends,
        "savingsProjection": savings_projection,
        "categoryBreakdown": category_breakdown,
        "expenses": expenses,
        "income": income,
        "monthly_expenses": monthly_expenses,
        "monthly_income": monthly_income,
        "category_expenses": category_expenses
    }

def create_charts(analysis_data: Dict) -> Dict[str, str]:
    """Create charts and return them as base64 encoded strings"""
    logger.info("Generating charts")
    
    charts = {}
    
    try:
        # Extract data
        expenses = analysis_data["expenses"]
        income = analysis_data["income"]
        monthly_expenses = analysis_data["monthly_expenses"]
        monthly_income = analysis_data["monthly_income"]
        category_expenses = analysis_data["category_expenses"]
        
        # 1. Pie Chart - Expense Distribution by Category
        plt.figure(figsize=(8, 8))
        plt.clf()
        
        # Get top categories and group small ones as "Other"
        top_n = 7
        if len(category_expenses) > top_n:
            top_categories = category_expenses.nlargest(top_n)
            other_sum = category_expenses[~category_expenses.index.isin(top_categories.index)].sum()
            
            if other_sum > 0:
                pie_data = top_categories.copy()
                pie_data["Other"] = other_sum
            else:
                pie_data = top_categories
        else:
            pie_data = category_expenses
        
        # Get colors for categories
        colors = []
        for category in pie_data.index:
            if category in CATEGORY_COLORS:
                colors.append(CATEGORY_COLORS[category])
            else:
                colors.append(CATEGORY_COLORS["Other"])
        
        # Create pie chart with nice styling
        plt.pie(
            pie_data, 
            labels=pie_data.index, 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
            textprops={'fontsize': 12, 'fontweight': 'bold'},
            shadow=False
        )
        
        plt.title("Expense Distribution by Category", fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        charts["pieChart"] = base64.b64encode(buffer.getvalue()).decode()
        
        # 2. Line Chart - Income vs Expenses over time
        plt.figure(figsize=(10, 6))
        plt.clf()
        
        # Convert period index to string for plotting
        if len(monthly_expenses) > 0 and len(monthly_income) > 0:
            # Get last 12 months or all if less
            months_to_show = min(12, len(monthly_expenses))
            
            expense_data = monthly_expenses.tail(months_to_show)
            income_data = monthly_income.tail(months_to_show)
            
            # Create a common index with all months
            all_months = sorted(set(expense_data.index) | set(income_data.index))
            
            # Convert to strings for plotting
            month_labels = [m.strftime('%b %Y') for m in all_months]
            
            # Get values for each series, filling NaN with 0
            expense_values = [expense_data.get(m, 0) for m in all_months]
            income_values = [income_data.get(m, 0) for m in all_months]
            
            # Calculate net savings
            savings_values = [i - e for i, e in zip(income_values, expense_values)]
            
            # Plot with nice styling
            plt.plot(month_labels, income_values, marker='o', linewidth=3, label='Income', color='#38b000')
            plt.plot(month_labels, expense_values, marker='s', linewidth=3, label='Expenses', color='#ff006e')
            plt.plot(month_labels, savings_values, marker='^', linewidth=3, label='Net Savings', color='#3a86ff')
            
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title("Monthly Income vs Expenses", fontsize=16, fontweight='bold', pad=20)
            plt.xlabel("Month", fontsize=12, fontweight='bold')
            plt.ylabel("Amount (₹)", fontsize=12, fontweight='bold')
            plt.xticks(rotation=45)
            plt.legend(fontsize=12)
            
            # Add some styling
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add data labels
            for i, (inc, exp, sav) in enumerate(zip(income_values, expense_values, savings_values)):
                plt.annotate(f'{inc:,.0f}', (i, inc), textcoords="offset points", 
                             xytext=(0,10), ha='center', fontsize=9, fontweight='bold', color='#38b000')
                plt.annotate(f'{exp:,.0f}', (i, exp), textcoords="offset points", 
                             xytext=(0,-15), ha='center', fontsize=9, fontweight='bold', color='#ff006e')
            
            plt.tight_layout()
            
            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            charts["lineChart"] = base64.b64encode(buffer.getvalue()).decode()
        else:
            # Create empty chart if no data
            plt.title("No monthly data available", fontsize=16)
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            plt.close()
            charts["lineChart"] = base64.b64encode(buffer.getvalue()).decode()
        
        # 3. Bar Chart - Top Expense Categories
        plt.figure(figsize=(10, 6))
        plt.clf()
        
        # Get top categories
        top_categories = category_expenses.nlargest(5)
        
        # Create bar chart with nice styling
        bars = plt.bar(
            top_categories.index, 
            top_categories.values,
            color=CHART_COLORS["accent"][:len(top_categories)],
            edgecolor='white',
            linewidth=1.5
        )
        
        plt.title("Top Expense Categories", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Category", fontsize=12, fontweight='bold')
        plt.ylabel("Amount (₹)", fontsize=12, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'₹{height:,.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add some styling
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        charts["barChart"] = base64.b64encode(buffer.getvalue()).decode()
        
    except Exception as e:
        logger.error(f"Error generating charts: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create empty charts if there's an error
        for chart_type in ["pieChart", "lineChart", "barChart"]:
            if chart_type not in charts:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, f"Error generating {chart_type}", 
                         horizontalalignment='center', verticalalignment='center',
                         fontsize=14)
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100)
                plt.close()
                charts[chart_type] = base64.b64encode(buffer.getvalue()).decode()
    
    return charts

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    """Analyze financial data from CSV and return insights and visualizations"""
    try:
        logger.info(f"Received analysis request: {request}")
        
        # Load data from CSV or generate sample data
        if request.sampleData:
            df = generate_sample_data()
        elif request.csvUrl:
            df = await load_data_from_csv(request.csvUrl)
        else:
            logger.warning("No CSV URL provided and sample data not requested")
            raise HTTPException(status_code=400, detail="Either csvUrl or sampleData must be provided")
        
        # Filter data by time range if specified
        df = filter_data_by_time_range(df, request.timeRange or "all")
        
        # Filter by categories if specified
        if request.categories:
            df = df[df['category'].isin(request.categories)]
        
        # Analyze the data
        analysis_results = analyze_financial_data(df)
        
        # Generate charts
        charts = create_charts(analysis_results)
        
        # Prepare response
        response = {
            "insights": analysis_results["insights"],
            "recommendations": analysis_results["recommendations"],
            "budgetSuggestions": analysis_results["budgetSuggestions"],
            "spendingTrends": analysis_results["spendingTrends"],
            "savingsProjection": analysis_results["savingsProjection"],
            "categoryBreakdown": analysis_results["categoryBreakdown"],
            "charts": charts
        }
        
        logger.info("Analysis completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during analysis: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Financial Analysis AI",
        "version": "1.0.0",
        "description": "AI model for analyzing financial data and generating insights",
        "endpoints": {
            "/analyze": "POST - Analyze financial data from CSV",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the FastAPI app
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
