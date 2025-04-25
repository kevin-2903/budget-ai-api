import base64
import io
import json
import os
import random
import sys
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Create FastAPI app
app = FastAPI(title="NIDZO Financial Analysis API")

# Add CORS middleware to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up Seaborn style for better-looking charts
sns.set(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['figure.facecolor'] = '#ffffff'

# Define request and response models
class AnalysisRequest(BaseModel):
    csvUrl: Optional[str] = None
    sampleData: Optional[bool] = False

class AnalysisResponse(BaseModel):
    insights: str
    recommendations: str
    future_predictions: str
    charts: Dict[str, str]

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = f"Unhandled error: {str(exc)}"
    error_trace = traceback.format_exc()
    print(f"ERROR: {error_msg}\n{error_trace}")
    return JSONResponse(
        status_code=500,
        content={"detail": error_msg, "traceback": error_trace},
    )

# Helper function to generate sample data if no CSV is provided
def generate_sample_data() -> pd.DataFrame:
    """Generate sample financial data for demonstration purposes."""
    try:
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Generate dates for the past 6 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Categories with their probabilities and typical amounts
        expense_categories = {
            'Food & Dining': {'prob': 0.3, 'mean': 500, 'std': 200},
            'Transportation': {'prob': 0.15, 'mean': 300, 'std': 100},
            'Entertainment': {'prob': 0.1, 'mean': 800, 'std': 300},
            'Housing': {'prob': 0.05, 'mean': 10000, 'std': 1000},
            'Utilities': {'prob': 0.1, 'mean': 2000, 'std': 500},
            'Shopping': {'prob': 0.2, 'mean': 1000, 'std': 500},
            'Healthcare': {'prob': 0.05, 'mean': 1500, 'std': 1000},
            'Other': {'prob': 0.05, 'mean': 500, 'std': 200}
        }
        
        income_categories = {
            'Salary': {'prob': 0.7, 'mean': 45000, 'std': 5000},
            'Freelance': {'prob': 0.1, 'mean': 10000, 'std': 5000},
            'Investments': {'prob': 0.1, 'mean': 5000, 'std': 2000},
            'Other': {'prob': 0.1, 'mean': 3000, 'std': 1000}
        }
        
        # Generate transactions
        transactions = []
        
        # Generate random expenses
        for _ in range(100):
            date = np.random.choice(dates)
            category = np.random.choice(
                list(expense_categories.keys()), 
                p=[cat['prob'] for cat in expense_categories.values()]
            )
            cat_params = expense_categories[category]
            amount = max(100, np.random.normal(cat_params['mean'], cat_params['std']))
            
            transactions.append({
                'id': f"exp-{len(transactions)}",
                'type': 'expense',
                'date': date.strftime('%Y-%m-%d'),
                'amount': round(amount, 2),
                'category': category,
                'description': f"{category} expense"
            })
        
        # Generate random income (fewer, larger transactions)
        for _ in range(10):
            date = np.random.choice(dates)
            category = np.random.choice(
                list(income_categories.keys()), 
                p=[cat['prob'] for cat in income_categories.values()]
            )
            cat_params = income_categories[category]
            amount = max(1000, np.random.normal(cat_params['mean'], cat_params['std']))
            
            transactions.append({
                'id': f"inc-{len(transactions)}",
                'type': 'income',
                'date': date.strftime('%Y-%m-%d'),
                'amount': round(amount, 2),
                'category': category,
                'description': f"{category} income"
            })
        
        # Convert to DataFrame and sort by date
        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        return df
    except Exception as e:
        print(f"Error generating sample data: {e}")
        # Create a minimal valid dataframe as fallback
        return pd.DataFrame({
            'id': ['exp-1', 'inc-1'],
            'type': ['expense', 'income'],
            'date': [datetime.now(), datetime.now()],
            'amount': [1000, 5000],
            'category': ['Other', 'Salary'],
            'description': ['Fallback expense', 'Fallback income']
        })

# Function to load data from CSV URL
def load_data_from_csv(csv_url: str) -> pd.DataFrame:
    """Load financial data from a CSV URL."""
    try:
        # Try to load data from the URL
        df = pd.read_csv(csv_url)
        print(f"Successfully loaded CSV from URL with {len(df)} rows")
        
        # Ensure required columns exist
        required_columns = ['id', 'type', 'date', 'amount', 'category']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: CSV is missing required columns: {missing_columns}")
            print("Generating sample data instead.")
            return generate_sample_data()
        
        # Convert date to datetime
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as e:
            print(f"Error converting date column: {e}")
            df['date'] = pd.to_datetime('today')
        
        # Ensure type is either 'expense' or 'income'
        if not all(df['type'].isin(['expense', 'income'])):
            print("Warning: 'type' column should only contain 'expense' or 'income'")
            df['type'] = df['type'].map(lambda x: 'expense' if x != 'income' else 'income')
        
        # Ensure amount is numeric
        try:
            df['amount'] = pd.to_numeric(df['amount'])
        except Exception as e:
            print(f"Error converting amount column: {e}")
            df['amount'] = 1000  # Default value
        
        return df
    
    except Exception as e:
        print(f"Error loading CSV from URL: {e}")
        print("Generating sample data instead.")
        return generate_sample_data()

# Function to analyze financial data
def analyze_financial_data(df: pd.DataFrame) -> Dict:
    """Analyze financial data and return insights, recommendations, and predictions."""
    try:
        # Calculate basic metrics
        expenses = df[df['type'] == 'expense'].copy()
        income = df[df['type'] == 'income'].copy()
        
        # Handle empty dataframes
        if expenses.empty:
            print("Warning: No expense data found. Adding dummy expense.")
            expenses = pd.DataFrame({
                'id': ['exp-dummy'],
                'type': ['expense'],
                'date': [datetime.now()],
                'amount': [1000],
                'category': ['Other'],
                'description': ['Dummy expense']
            })
        
        if income.empty:
            print("Warning: No income data found. Adding dummy income.")
            income = pd.DataFrame({
                'id': ['inc-dummy'],
                'type': ['income'],
                'date': [datetime.now()],
                'amount': [5000],
                'category': ['Salary'],
                'description': ['Dummy income']
            })
        
        total_expenses = expenses['amount'].sum()
        total_income = income['amount'].sum()
        net_cashflow = total_income - total_expenses
        savings_rate = (net_cashflow / total_income * 100) if total_income > 0 else 0
        
        # Group by month for trend analysis
        expenses['month'] = expenses['date'].dt.to_period('M')
        income['month'] = income['date'].dt.to_period('M')
        
        monthly_expenses = expenses.groupby('month')['amount'].sum()
        monthly_income = income.groupby('month')['amount'].sum()
        
        # Calculate average monthly expense and income
        avg_monthly_expense = monthly_expenses.mean() if not monthly_expenses.empty else 0
        avg_monthly_income = monthly_income.mean() if not monthly_income.empty else 0
        
        # Analyze expense categories
        category_expenses = expenses.groupby('category')['amount'].sum().sort_values(ascending=False)
        top_expense_categories = category_expenses.head(3)
        
        # Generate insights
        insights = []
        insights.append(f"Total Income: ₹{total_income:.2f}")
        insights.append(f"Total Expenses: ₹{total_expenses:.2f}")
        insights.append(f"Net Cashflow: ₹{net_cashflow:.2f}")
        insights.append(f"Savings Rate: {savings_rate:.1f}%")
        insights.append(f"Average Monthly Expense: ₹{avg_monthly_expense:.2f}")
        
        # Add insights about top expense categories
        insights.append("\nTop Expense Categories:")
        for category, amount in top_expense_categories.items():
            percentage = (amount / total_expenses) * 100
            insights.append(f"{category}: ₹{amount:.2f} ({percentage:.1f}% of total)")
        
        # Add monthly breakdown if we have enough data
        if len(monthly_expenses) >= 3:
            insights.append("\nMonthly Breakdown:")
            for month in sorted(monthly_expenses.index)[-3:]:  # Last 3 months
                month_name = month.strftime('%B %Y')
                exp = monthly_expenses.get(month, 0)
                inc = monthly_income.get(month, 0)
                net = inc - exp
                insights.append(f"{month_name}: Income ₹{inc:.2f}, Expenses ₹{exp:.2f}, Net ₹{net:.2f}")
        
        # Generate recommendations
        recommendations = []
        
        # Savings recommendations
        if savings_rate < 20:
            recommendations.append("1. Increase Savings: Your current savings rate is below the recommended 20%. Consider reducing discretionary spending to increase your savings.")
        else:
            recommendations.append("1. Maintain Savings: You're saving at a good rate. Consider investing your surplus for long-term growth.")
        
        # Category-specific recommendations
        if not category_expenses.empty:
            highest_category = category_expenses.index[0]
            highest_amount = category_expenses.iloc[0]
            highest_percentage = (highest_amount / total_expenses) * 100
            
            if highest_percentage > 30:
                recommendations.append(f"2. Reduce {highest_category} Spending: This category represents {highest_percentage:.1f}% of your expenses. Look for ways to reduce this spending.")
            
            # Budget allocation recommendation
            recommendations.append("3. Budget Allocation: Consider using the 50/30/20 rule - 50% for needs, 30% for wants, and 20% for savings and debt repayment.")
        
        # Income recommendations
        if total_income > 0 and len(monthly_income) > 1:
            income_trend = monthly_income.pct_change().mean() * 100
            if income_trend < 0:
                recommendations.append("4. Income Growth: Your income shows a declining trend. Consider exploring additional income sources or negotiating a raise.")
        
        # Generate future predictions
        predictions = []
        
        # Predict next 3 months of expenses
        if len(monthly_expenses) >= 3:
            last_3_expenses = monthly_expenses[-3:].mean()
            for i in range(1, 4):
                next_month = (datetime.now() + timedelta(days=30*i)).strftime('%B %Y')
                # Add some randomness to predictions
                predicted_expense = last_3_expenses * (1 + (random.uniform(-0.1, 0.1)))
                predictions.append(f"{next_month}: ₹{predicted_expense:.2f}")
        else:
            # If not enough data, make simple predictions
            for i in range(1, 4):
                next_month = (datetime.now() + timedelta(days=30*i)).strftime('%B %Y')
                predicted_expense = avg_monthly_expense * (1 + (random.uniform(-0.1, 0.1)))
                predictions.append(f"{next_month}: ₹{predicted_expense:.2f}")
        
        # Add savings projection
        if savings_rate > 0:
            monthly_savings = avg_monthly_income - avg_monthly_expense
            predictions.append(f"\nProjected Monthly Savings: ₹{monthly_savings:.2f}")
            predictions.append(f"Projected Annual Savings: ₹{monthly_savings*12:.2f}")
        
        return {
            "insights": "\n".join(insights),
            "recommendations": "\n".join(recommendations),
            "future_predictions": "\n".join(predictions),
            "data": {
                "expenses": expenses.to_dict('records'),
                "income": income.to_dict('records'),
                "monthly_expenses": {str(k): float(v) for k, v in monthly_expenses.items()},
                "monthly_income": {str(k): float(v) for k, v in monthly_income.items()},
                "category_expenses": {k: float(v) for k, v in category_expenses.items()}
            }
        }
    except Exception as e:
        print(f"Error in analyze_financial_data: {e}")
        print(traceback.format_exc())
        
        # Return fallback data
        return {
            "insights": "Unable to generate detailed insights due to an error. Please try again later.",
            "recommendations": "1. Consider reviewing your financial data for accuracy.\n2. Ensure your income and expense records are properly categorized.",
            "future_predictions": "Future predictions are not available at this time.",
            "data": {
                "expenses": [],
                "income": [],
                "monthly_expenses": {},
                "monthly_income": {},
                "category_expenses": {"Other": 1000}
            }
        }

# Function to create charts
def create_charts(analysis_data: Dict) -> Dict[str, str]:
    """Create charts based on analysis data and return them as base64 encoded strings."""
    charts = {}
    
    try:
        # Extract data
        data = analysis_data.get("data", {})
        
        # Handle missing or empty data
        if not data:
            raise ValueError("No data available for chart generation")
        
        # Create expense dataframe
        expenses_data = data.get("expenses", [])
        if not expenses_data:
            expenses_data = [{"category": "Other", "amount": 1000, "date": datetime.now().isoformat()}]
        expenses = pd.DataFrame(expenses_data)
        
        # Create income dataframe
        income_data = data.get("income", [])
        if not income_data:
            income_data = [{"category": "Salary", "amount": 5000, "date": datetime.now().isoformat()}]
        income = pd.DataFrame(income_data)
        
        # Get category expenses
        category_expenses = data.get("category_expenses", {"Other": 1000})
        
        # 1. Pie Chart - Expense Distribution by Category
        plt.figure(figsize=(10, 6))
        plt.clf()
        
        # Use a colorful palette
        colors = sns.color_palette("husl", len(category_expenses))
        
        # Create pie chart
        plt.pie(
            list(category_expenses.values()),
            labels=list(category_expenses.keys()),
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1}
        )
        plt.title('Expense Distribution by Category', fontsize=16, pad=20)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        pie_chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        charts["pie_chart_base64"] = pie_chart_base64
        
        # 2. Line Chart - Monthly Income vs Expenses
        if not expenses.empty and not income.empty:
            plt.figure(figsize=(12, 6))
            plt.clf()
            
            try:
                # Ensure date column is datetime
                if 'date' in expenses.columns:
                    expenses['date'] = pd.to_datetime(expenses['date'])
                    expenses['month'] = expenses['date'].dt.to_period('M')
                else:
                    expenses['month'] = pd.Period(datetime.now(), freq='M')
                
                if 'date' in income.columns:
                    income['date'] = pd.to_datetime(income['date'])
                    income['month'] = income['date'].dt.to_period('M')
                else:
                    income['month'] = pd.Period(datetime.now(), freq='M')
                
                # Group by month
                monthly_expenses = expenses.groupby('month')['amount'].sum()
                monthly_income = income.groupby('month')['amount'].sum()
                
                # Create a date range for all months
                all_months = pd.period_range(
                    start=min(monthly_expenses.index.min(), monthly_income.index.min()),
                    end=max(monthly_expenses.index.max(), monthly_income.index.max()),
                    freq='M'
                )
                
                # Reindex to include all months
                monthly_expenses = monthly_expenses.reindex(all_months, fill_value=0)
                monthly_income = monthly_income.reindex(all_months, fill_value=0)
                
                # Plot
                plt.plot(
                    [str(m) for m in monthly_expenses.index],
                    monthly_expenses.values,
                    marker='o',
                    linestyle='-',
                    color='#FF6B6B',
                    linewidth=2,
                    label='Expenses'
                )
                plt.plot(
                    [str(m) for m in monthly_income.index],
                    monthly_income.values,
                    marker='s',
                    linestyle='-',
                    color='#4ECDC4',
                    linewidth=2,
                    label='Income'
                )
            except Exception as e:
                print(f"Error creating line chart: {e}")
                # Fallback to simple chart
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
                expenses_values = [20000, 22000, 19000, 23000, 21000, 24000]
                income_values = [30000, 30000, 32000, 28000, 35000, 33000]
                
                plt.plot(months, expenses_values, marker='o', linestyle='-', color='#FF6B6B', linewidth=2, label='Expenses')
                plt.plot(months, income_values, marker='s', linestyle='-', color='#4ECDC4', linewidth=2, label='Income')
            
            plt.title('Monthly Income vs Expenses', fontsize=16, pad=20)
            plt.xlabel('Month', fontsize=12)
            plt.ylabel('Amount (₹)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            line_chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            charts["line_chart_base64"] = line_chart_base64
        
        # 3. Bar Chart - Top Categories Comparison
        plt.figure(figsize=(12, 6))
        plt.clf()
        
        # Get top categories
        top_categories = list(category_expenses.keys())[:5]  # Top 5 categories
        top_amounts = list(category_expenses.values())[:5]
        
        # Create bar chart
        bars = plt.bar(
            top_categories,
            top_amounts,
            color=sns.color_palette("husl", len(top_categories)),
            edgecolor='white',
            linewidth=1
        )
        
        # Add data labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.1,
                f'₹{int(height):,}',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        plt.title('Top Expense Categories', fontsize=16, pad=20)
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Amount (₹)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        bar_chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        charts["bar_chart_base64"] = bar_chart_base64
        
    except Exception as e:
        print(f"Error creating charts: {e}")
        print(traceback.format_exc())
        
        # Generate fallback charts
        try:
            # Create a simple fallback pie chart
            plt.figure(figsize=(10, 6))
            plt.clf()
            plt.pie(
                [60, 30, 10],
                labels=['Category A', 'Category B', 'Category C'],
                autopct='%1.1f%%',
                colors=['#FF6B6B', '#4ECDC4', '#C7F464']
            )
            plt.title('Expense Distribution (Fallback Chart)', fontsize=16)
            plt.axis('equal')
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            fallback_chart = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            charts["pie_chart_base64"] = fallback_chart
            charts["line_chart_base64"] = fallback_chart
            charts["bar_chart_base64"] = fallback_chart
        except Exception as e:
            print(f"Error creating fallback charts: {e}")
            # If even fallback charts fail, return empty strings
            charts["pie_chart_base64"] = ""
            charts["line_chart_base64"] = ""
            charts["bar_chart_base64"] = ""
    
    return charts

# API endpoint for analysis
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    """Analyze financial data and return insights, recommendations, and visualizations."""
    try:
        print(f"Received analysis request: {request}")
        
        # Load data
        if request.csvUrl:
            print(f"Loading data from CSV URL: {request.csvUrl}")
            df = load_data_from_csv(request.csvUrl)
        else:
            print("No CSV URL provided, generating sample data")
            df = generate_sample_data()
        
        print(f"Data loaded successfully with {len(df)} records")
        
        # Analyze data
        print("Analyzing financial data...")
        analysis_results = analyze_financial_data(df)
        
        # Create charts
        print("Generating charts...")
        charts = create_charts(analysis_results)
        
        # Return results
        print("Analysis complete, returning results")
        return {
            "insights": analysis_results["insights"],
            "recommendations": analysis_results["recommendations"],
            "future_predictions": analysis_results["future_predictions"],
            "charts": charts
        }
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in analyze endpoint: {e}")
        print(error_trace)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Simple health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Root endpoint with basic info
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "NIDZO Financial Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "POST - Analyze financial data",
            "/health": "GET - Health check"
        }
    }

# Run the server if this file is executed directly
if __name__ == "__main__":
    # Check if running in development or production
    port = int(os.environ.get("PORT", 8000))
    
    print(f"Starting NIDZO Financial Analysis API on port {port}")
    print(f"Python version: {sys.version}")
    print(f"Pandas version: {pd.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Matplotlib version: {matplotlib.__version__}")
    
    # In production (like Render), we should bind to 0.0.0.0
    if os.environ.get("ENVIRONMENT") == "production":
        uvicorn.run("main:app", host="0.0.0.0", port=port)
    else:
        # For local development
        uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)
