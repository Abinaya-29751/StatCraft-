
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import os
import json
import math
from datetime import datetime, date
from io import BytesIO
from dotenv import load_dotenv
import base64
import tempfile


load_dotenv()

app = FastAPI(title="Dataset Analyzer")

# IMPORTANT: Configure upload limits
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Increase max request size
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware

class MaxSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Allow large uploads
        return await call_next(request)

app.add_middleware(MaxSizeMiddleware)

# Rest of your code...
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Load environment variables
load_dotenv()

app = FastAPI(title="Dataset Analyzer")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Gemini API key loaded successfully!")
else:
    print("WARNING: GOOGLE_API_KEY not found in environment variables!")

# HELPER FUNCTION - Convert to JSON-serializable format
def convert_to_json_serializable(obj):
    """Convert pandas/numpy types to JSON-serializable types"""
    if pd.isna(obj):
        return None
    elif isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.strftime('%Y-%m-%d')
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bool):
        return bool(obj)
    else:
        return str(obj)

# HELPER FUNCTION - Make values JSON-safe
def safe_float(value):
    """Convert value to JSON-safe float"""
    if pd.isna(value) or value is None:
        return None
    if math.isinf(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

# HOME ROUTE
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# AI INSIGHTS ROUTE
@app.get("/ai-insights", response_class=HTMLResponse)
async def ai_insights(request: Request):
    return templates.TemplateResponse("ai_insights.html", {"request": request})

# STATISTICAL ANALYSIS ROUTE
@app.get("/statistical-analysis", response_class=HTMLResponse)
async def statistical_analysis(request: Request):
    return templates.TemplateResponse("statistical_analysis.html", {"request": request})

# VISUALIZATIONS ROUTE
@app.get("/visualizations", response_class=HTMLResponse)
async def visualizations(request: Request):
    return templates.TemplateResponse("visualizations.html", {"request": request})

# DATA PREVIEW ROUTE
@app.get("/data-preview", response_class=HTMLResponse)
async def data_preview(request: Request):
    return templates.TemplateResponse("data_preview.html", {"request": request})

# HEALTH CHECK ROUTE
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "API is running",
        "ai_provider": "Google Gemini"
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        print(f"Processing: {file.filename}")
        
        # Read file in chunks
        max_size = 200 * 1024 * 1024
        contents = bytearray()
        total_read = 0
        chunk_size = 10 * 1024 * 1024
        
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total_read += len(chunk)
            if total_read > max_size:
                raise HTTPException(status_code=413, detail="File exceeds 200MB")
            contents.extend(chunk)
        
        print(f"Loaded: {total_read/(1024*1024):.1f}MB")
        
        # Parse file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(bytes(contents)))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(BytesIO(bytes(contents)))
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
        
        #  CLEAN COLUMN NAMES
        print(f" Original columns: {list(df.columns)}")
        df.columns = df.columns.str.strip()  # Remove spaces
        print(f" Cleaned columns: {list(df.columns)}")
        
        print(f" Parsed: {len(df):,} rows × {len(df.columns)} columns")
        
        # Clean data
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Sample if needed
        original_rows = len(df)
        if original_rows > 100000:
            print(f" Sampling 50K from {original_rows:,} rows")
            df = df.sample(n=50000, random_state=42)
            sampled = True
        else:
            sampled = False
        
        print(" Generating comprehensive report...")
        
        # Generate analysis
        analysis = await analyze_dataset(df)
        
        if sampled:
            analysis['sampling_note'] = f"Analysis performed on 50,000 representative rows from {original_rows:,} total records"
        
        print(" Report complete!")
        
        return JSONResponse(content=analysis)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ANALYSIS FUNCTION
async def analyze_dataset(df):
    """Main analysis function with proper JSON serialization"""
    
    # Generate statistical analysis
    stats = generate_statistics(df)
    
    # Generate visualizations
    charts = generate_visualizations(df)
    
    # Generate AI-powered insights using Gemini
    insights = await generate_ai_insights(df, stats)
    
    # Convert preview data to JSON-safe format
    preview_df = df.head(10).copy()
    
    # Convert all columns to JSON-serializable format
    preview_records = []
    for _, row in preview_df.iterrows():
        record = {}
        for col in preview_df.columns:
            record[col] = convert_to_json_serializable(row[col])
        preview_records.append(record)
    
    # Convert column names to strings
    columns = [str(col) for col in df.columns.tolist()]
    
    return {
        "statistics": stats,
        "charts": charts,
        "insights": insights,
        "preview": preview_records,
        "columns": columns
    }

# STATISTICS FUNCTION
def generate_statistics(df):
    """Calculate statistical metrics with JSON-safe values"""
    stats = {}
    
    # Get numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate statistics for each numerical column
    for col in numerical_cols:
        col_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        
        stats[str(col)] = {
            "mean": safe_float(col_data.mean()) if len(col_data) > 0 else None,
            "median": safe_float(col_data.median()) if len(col_data) > 0 else None,
            "mode": safe_float(col_data.mode()[0]) if len(col_data.mode()) > 0 else None,
            "std": safe_float(col_data.std()) if len(col_data) > 0 else None,
            "min": safe_float(col_data.min()) if len(col_data) > 0 else None,
            "max": safe_float(col_data.max()) if len(col_data) > 0 else None,
            "missing": int(df[col].isnull().sum()),
            "count": int(len(col_data))
        }
    
    # Dataset overview
    stats["overview"] = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "numerical_columns": len(numerical_cols),
        "categorical_columns": len(df.select_dtypes(include=['object']).columns),
        "total_missing_values": int(df.isnull().sum().sum())
    }
    
    return stats

def generate_visualizations(df):
    """Generate comprehensive charts including bar, line, histogram, pie, and correlation heatmap"""
    charts = []
    
    # Clean data
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    if len(numerical_cols) == 0:
        return charts
    
    # 1. BAR CHART - Top values
    try:
        if len(numerical_cols) > 0:
            col = numerical_cols[0]
            data = df_clean[col].dropna().head(10)
            
            if len(data) > 0 and data.std() > 0:
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(range(len(data))),
                        y=data.values.tolist(),
                        marker_color='rgb(102, 126, 234)',
                        text=data.values.round(2),
                        textposition='outside'
                    )
                ])
                fig.update_layout(
                    title=f"Bar Chart - {col}",
                    xaxis_title="Index",
                    yaxis_title=col,
                    height=400
                )
                charts.append({
                    "type": "bar",
                    "title": f"Bar Chart - {col}",
                    "data": fig.to_json()
                })
    except Exception as e:
        print(f"Bar chart error: {e}")
    
    # 2. LINE CHART - Trends
    try:
        if len(numerical_cols) >= 1:
            col = numerical_cols[0]
            data = df_clean[col].dropna().head(30)
            
            if len(data) > 1:
                fig = go.Figure(data=[
                    go.Scatter(
                        x=list(range(len(data))),
                        y=data.values.tolist(),
                        mode='lines+markers',
                        line=dict(color='rgb(118, 75, 162)', width=2),
                        marker=dict(size=8)
                    )
                ])
                fig.update_layout(
                    title=f"Line Chart - Trend Analysis ({col})",
                    xaxis_title="Record Index",
                    yaxis_title=col,
                    height=400
                )
                charts.append({
                    "type": "line",
                    "title": f"Line Chart - {col}",
                    "data": fig.to_json()
                })
    except Exception as e:
        print(f"Line chart error: {e}")
    
    # 3. PIE CHART - Distribution by Category
    try:
        if len(categorical_cols) > 0:
            # Find first categorical column
            cat_col = categorical_cols[0]
            
            # Get value counts
            value_counts = df_clean[cat_col].value_counts().head(8)
            
            if len(value_counts) > 0:
                fig = go.Figure(data=[
                    go.Pie(
                        labels=value_counts.index.tolist(),
                        values=value_counts.values.tolist(),
                        hole=0.3,
                        marker=dict(
                            colors=['#667eea', '#764ba2', '#f093fb', '#4facfe', 
                                   '#43e97b', '#fa709a', '#fee140', '#30cfd0']
                        )
                    )
                ])
                fig.update_layout(
                    title=f"Distribution by {cat_col}",
                    height=450
                )
                charts.append({
                    "type": "pie",
                    "title": f"Pie Chart - {cat_col} Distribution",
                    "data": fig.to_json()
                })
    except Exception as e:
        print(f"Pie chart error: {e}")
    
    # 4. HISTOGRAM - Distribution
    try:
        if len(numerical_cols) > 0:
            col = numerical_cols[0]
            data = df_clean[col].dropna()
            
            if len(data) > 1 and data.nunique() > 1:
                fig = px.histogram(
                    x=data.values.tolist(),
                    title=f"Distribution - {col}",
                    labels={'x': col, 'y': 'Frequency'},
                    nbins=min(30, int(len(data) / 10)),
                    color_discrete_sequence=['#667eea']
                )
                fig.update_layout(
                    xaxis_title=col,
                    yaxis_title="Frequency",
                    height=400
                )
                charts.append({
                    "type": "histogram",
                    "title": f"Distribution - {col}",
                    "data": fig.to_json()
                })
    except Exception as e:
        print(f"Histogram error: {e}")
    
    # 5. PIE CHART - Numerical Ranges (if applicable)
    try:
        if len(numerical_cols) >= 2:
            col = numerical_cols[1]
            col_data = df_clean[col].dropna()
            
            if len(col_data) > 10:
                # Create range bins
                q1, q2, q3 = col_data.quantile([0.25, 0.5, 0.75])
                
                bins = [
                    ('Low', col_data[col_data <= q1].count()),
                    ('Medium', col_data[(col_data > q1) & (col_data <= q3)].count()),
                    ('High', col_data[col_data > q3].count())
                ]
                
                labels, values = zip(*[(b[0], b[1]) for b in bins if b[1] > 0])
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        marker=dict(colors=['#4facfe', '#667eea', '#764ba2'])
                    )
                ])
                fig.update_layout(
                    title=f"{col} - Range Distribution",
                    height=400
                )
                charts.append({
                    "type": "pie",
                    "title": f"Pie Chart - {col} Ranges",
                    "data": fig.to_json()
                })
    except Exception as e:
        print(f"Pie chart (ranges) error: {e}")
    
    # 6. CORRELATION HEATMAP
    try:
        if len(numerical_cols) >= 2:
            corr_data = df_clean[numerical_cols[:8]].dropna()
            
            if len(corr_data) >= 3:
                corr_matrix = corr_data.corr()
                
                if not corr_matrix.isnull().all().all():
                    corr_matrix = corr_matrix.fillna(0)
                    
                    fig = px.imshow(
                        corr_matrix,
                        title="Correlation Heatmap",
                        labels=dict(color="Correlation"),
                        color_continuous_scale="RdBu",
                        aspect="auto",
                        text_auto=".2f"
                    )
                    fig.update_layout(height=500)
                    charts.append({
                        "type": "heatmap",
                        "title": "Correlation Heatmap",
                        "data": fig.to_json()
                    })
    except Exception as e:
        print(f"Heatmap error: {e}")
    
    if len(charts) == 0:
        print("No visualizations generated")
    
    return charts

async def generate_ai_insights(df, stats):
    """Generate AI-powered insights focused on trends, patterns, and anomalies"""
    
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")
    
    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
        # Get dataset info
        total_rows = stats['overview']['total_rows']
        total_cols = stats['overview']['total_columns']
        numerical_cols = stats['overview']['numerical_columns']
        missing_vals = stats['overview']['total_missing_values']
        
        # Get sample data
        sample_data = []
        for _, row in df.head(3).iterrows():
            row_dict = {}
            for col in df.columns:
                row_dict[col] = convert_to_json_serializable(row[col])
            sample_data.append(row_dict)
        
        # Get key statistics for trends
        top_stats = {}
        for col in list(stats.keys())[:5]:
            if col != 'overview':
                top_stats[col] = {
                    'mean': stats[col].get('mean'),
                    'median': stats[col].get('median'),
                    'std': stats[col].get('std'),
                    'min': stats[col].get('min'),
                    'max': stats[col].get('max')
                }
        
        # Enhanced prompt for trends and patterns
        prompt = f"""You are a data analyst expert. Analyze this dataset and provide a comprehensive textual summary focusing specifically on TRENDS, PATTERNS, and ANOMALIES.

        **DATASET OVERVIEW:**
        - Total Records: {total_rows:,}
        - Total Attributes: {total_cols}
        - Numerical Metrics: {numerical_cols}
        - Data Completeness: {((total_rows * total_cols - missing_vals) / (total_rows * total_cols) * 100):.1f}%

        **KEY STATISTICS:**
        {json.dumps(top_stats, indent=2)}

        **SAMPLE DATA (first 3 records):**
        {json.dumps(sample_data, indent=2)}

        **GENERATE A DETAILED TEXTUAL SUMMARY WITH THESE SECTIONS:**

        ## Executive Summary
        Brief overview of the dataset and its purpose (2-3 sentences).

        ## Key Trends Identified
        Identify 4-5 major trends in the data:
        - What patterns emerge across different metrics?
        - Are values increasing, decreasing, or stable?
        - Are there seasonal or cyclical patterns?
        - What correlations exist between variables?

        ## Notable Patterns
        Describe 3-4 significant patterns:
        - Distribution patterns (normal, skewed, bimodal)
        - Clustering of values
        - Relationships between categories and metrics
        - Consistent behaviors across segments

        ## Anomalies and Outliers
        Identify potential anomalies:
        - Unusual values or outliers
        - Data quality issues
        - Inconsistencies or gaps
        - Values that deviate significantly from the norm

        ## Statistical Insights
        - Mean vs median analysis (what does the difference tell us?)
        - Standard deviation interpretation (high variability?)
        - Range analysis (min-max spread)
        - Data distribution characteristics

        ## Data Quality Observations
        - Missing values impact
        - Completeness assessment
        - Reliability of insights

        ## Actionable Recommendations
        Provide 5 specific, data-driven recommendations based on the trends and patterns identified.

        **FORMAT RULES:**
        - Use ## for main headers
        - Use bullet points with * for lists
        - Use **bold** for emphasis
        - Reference actual numbers from the statistics
        - Keep total response 400-500 words
        - Be specific and quantitative"""

        # Safety settings
        safety = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Try models
        models = ['models/gemini-2.0-flash-exp', 'models/gemini-1.5-pro', 'gemini-pro']
        
        for model_name in models:
            try:
                print(f" Generating insights with {model_name}...")
                
                model = genai.GenerativeModel(model_name)
                
                response = model.generate_content(
                    prompt,
                    generation_config={'temperature': 0.7, 'max_output_tokens': 2048},
                    safety_settings=safety
                )
                
                if response and response.text:
                    print(f"Insights generated!")
                    
                    # Add header and footer
                    report = generate_report_header(df, stats)
                    report += "\n\n---\n\n"
                    report += "## AI-Generated Analysis: Trends, Patterns & Anomalies\n\n"
                    report += response.text
                    report += "\n\n"
                    
                    return report
                    
            except Exception as e:
                print(f" {model_name} failed: {str(e)}")
                continue
        
        # Fallback
        return generate_structured_report(df, stats)
        
    except Exception as e:
        print(f" Error: {str(e)}")
        return generate_structured_report(df, stats)

def generate_report_header(df, stats):
    """Generate professional report header"""
    from datetime import datetime
    
    current_date = datetime.now().strftime("%B %d, %Y")
    
    header = f"""# StatCraft INSIGHTS REPORT

**Prepared by:** StatCraft
**Generated on:** {current_date}  
**Dataset Source:** Uploaded File  
---

## DATASET PROFILE

    **Total Records:** {stats['overview']['total_rows']:,}  
    **Total Attributes:** {stats['overview']['total_columns']}  
    **Numerical Columns:** {stats['overview']['numerical_columns']}  
    **Categorical Columns:** {stats['overview']['categorical_columns']}  
    **Missing Values:** {stats['overview']['total_missing_values']}  
    **Data Completeness:** {((stats['overview']['total_rows'] * stats['overview']['total_columns'] - stats['overview']['total_missing_values']) / (stats['overview']['total_rows'] * stats['overview']['total_columns']) * 100):.1f}%

    **Column Names**
    {', '.join([f'`{col}`' for col in df.columns[:15]])}{'...' if len(df.columns) > 15 else ''}
    
    """
        
    return header

def generate_report_footer(df):
    """Generate report appendix"""
    
    
    footer = f"""

## REPORT NOTES

- This report was automatically generated using AI-powered analysis
- Statistics are calculated from the complete dataset
- Large datasets (>100K rows) are sampled for performance optimization
- All insights are data-driven and based on statistical patterns

**Report End** | Generated by Automated Dataset Analyzer | Powered by Google Gemini AI"""
    
    return footer

# HEALTH CHECK ENDPOINT
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "message": "API is running",
        "ai_provider": "Google Gemini"
    }

# TEST API KEY ENDPOINT
@app.get("/test-api-key")
async def test_api_key():
    """Test if Gemini API key is loaded"""
    if GOOGLE_API_KEY:
        masked_key = GOOGLE_API_KEY[:10] + "..." + GOOGLE_API_KEY[-4:]
        return {
            "status": "success",
            "message": "Gemini API key is loaded",
            "key_preview": masked_key,
            "provider": "Google Gemini"
        }
    else:
        return {
            "status": "error",
            "message": "GOOGLE_API_KEY not found in environment"
        }

@app.get("/list-models")
async def list_available_models():
    """List all available Gemini models"""
    try:
        models = genai.list_models()
        available = []
        for model in models:
            available.append({
                "name": model.name,
                "supported_methods": model.supported_generation_methods
            })
        return {"status": "success", "models": available}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/download-report")
async def download_report(request: Request):
    """Generate and download comprehensive report"""
    try:
        # Get data from request body
        data = await request.json()
        
        if not data:
            raise HTTPException(status_code=400, detail="No data provided")
        
        # Generate comprehensive report
        report = generate_comprehensive_report(data)
        
        # Return report as HTML with embedded charts
        from fastapi.responses import HTMLResponse
        from datetime import datetime
        
        # Generate HTML report with charts
        html_report = generate_html_report(data)
        
        filename = f"dataset_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        return HTMLResponse(
            content=html_report,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        print(f"Download report error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.post("/download-pdf")
async def download_txt(request: Request):
    """Generate and download TXT report"""
    try:
        # Get data from request body
        data = await request.json()
        
        if not data:
            raise HTTPException(status_code=400, detail="No data provided")
        
        # Generate TXT report
        txt_content = generate_txt_report(data)
        
        filename = f"dataset_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        return Response(
            content=txt_content,
            media_type='text/plain',
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        print(f"Download TXT error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating TXT: {str(e)}")

def generate_html_report(data):
    """Generate HTML report with embedded charts"""
    try:
        stats = data.get('statistics', {})
        insights = data.get('insights', '')
        charts = data.get('charts', [])
        preview = data.get('preview', [])
        columns = data.get('columns', [])
        
        from datetime import datetime
        
        # Clean up insights
        cleaned_insights = insights
        if "AUTOMATED DATA INSIGHTS REPORT" in cleaned_insights:
            if "AI-Generated Analysis:" in cleaned_insights:
                parts = cleaned_insights.split("AI-Generated Analysis:")
                if len(parts) > 1:
                    cleaned_insights = "AI-Generated Analysis:" + parts[1]
            elif "Executive Summary" in cleaned_insights:
                parts = cleaned_insights.split("Executive Summary")
                if len(parts) > 1:
                    cleaned_insights = "Executive Summary" + parts[1]
        
        # Clean markdown formatting
        cleaned_insights = cleaned_insights.replace('# ', '').replace('## ', '').replace('### ', '')
        cleaned_insights = cleaned_insights.replace('**', '').replace('*', '').replace('---', '')
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }}
        .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .section h2 {{ color: #333; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }}
        .stat-value {{ font-size: 1.5em; font-weight: bold; color: #007bff; }}
        .chart-container {{ margin: 20px 0; padding: 15px; border: 1px solid #eee; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .insights {{ background: #f8f9fa; padding: 20px; border-radius: 5px; white-space: pre-line; }}
    </style>
</head>
<body>
    <div class="header">
        <h1> Dataset Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
    </div>
"""
        
        # Dataset Overview
        if stats and stats.get('overview'):
            overview = stats['overview']
            html_content += f"""
    <div class="section">
        <h2> Dataset Overview</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{overview.get('total_rows', 0):,}</div>
                <div>Total Rows</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{overview.get('total_columns', 0)}</div>
                <div>Total Columns</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{overview.get('numerical_columns', 0)}</div>
                <div>Numerical Columns</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{overview.get('total_missing_values', 0)}</div>
                <div>Missing Values</div>
            </div>
        </div>
    </div>
"""
        
        # AI Insights
        if insights:
            html_content += f"""
    <div class="section">
        <h2> AI-Generated Insights</h2>
        <div class="insights">{cleaned_insights}</div>
    </div>
"""
        
        # Statistical Analysis
        if stats:
            html_content += """
    <div class="section">
        <h2> Statistical Analysis</h2>
        <table>
            <thead>
                <tr>
                    <th>Column</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Missing</th>
                </tr>
            </thead>
            <tbody>
"""
            
            for column, values in stats.items():
                if column != 'overview' and isinstance(values, dict):
                    mean_val = values.get('mean')
                    median_val = values.get('median')
                    std_val = values.get('std')
                    min_val = values.get('min')
                    max_val = values.get('max')
                    
                    mean_str = f"{mean_val:.2f}" if mean_val is not None else 'N/A'
                    median_str = f"{median_val:.2f}" if median_val is not None else 'N/A'
                    std_str = f"{std_val:.2f}" if std_val is not None else 'N/A'
                    min_str = f"{min_val:.2f}" if min_val is not None else 'N/A'
                    max_str = f"{max_val:.2f}" if max_val is not None else 'N/A'
                    
                    html_content += f"""
                <tr>
                    <td><strong>{column}</strong></td>
                    <td>{mean_str}</td>
                    <td>{median_str}</td>
                    <td>{std_str}</td>
                    <td>{min_str}</td>
                    <td>{max_str}</td>
                    <td>{values.get('missing', 0)}</td>
                </tr>
"""
            
            html_content += """
            </tbody>
        </table>
    </div>
"""
        
        # Visualizations
        if charts:
            html_content += """
    <div class="section">
        <h2> Visualizations</h2>
"""
            
            for i, chart in enumerate(charts):
                chart_title = chart.get('title', f'Chart {i+1}')
                html_content += f"""
        <div class="chart-container">
            <h3>{chart_title}</h3>
            <div id="chart{i}"></div>
        </div>
"""
            
            # Add chart data as JSON
            html_content += """
        <script>
"""
            for i, chart in enumerate(charts):
                chart_data = chart.get('data', '{}')
                html_content += f"""
            try {{
                const plotData{i} = {chart_data};
                Plotly.newPlot('chart{i}', plotData{i}.data, plotData{i}.layout, {{responsive: true}});
            }} catch (error) {{
                document.getElementById('chart{i}').innerHTML = '<p>Error loading chart: ' + error.message + '</p>';
            }}
"""
            
            html_content += """
        </script>
    </div>
"""
        
        # Data Preview
        if preview and columns:
            html_content += """
    <div class="section">
        <h2> Data Preview (First 10 Rows)</h2>
        <table>
            <thead>
                <tr>
"""
            for col in columns:
                html_content += f"<th>{col}</th>"
            
            html_content += """
                </tr>
            </thead>
            <tbody>
"""
            
            for row in preview:
                html_content += "<tr>"
                for col in columns:
                    value = row.get(col, '-')
                    html_content += f"<td>{value}</td>"
                html_content += "</tr>"
            
            html_content += """
            </tbody>
        </table>
    </div>
"""
        
        html_content += """
    <div class="section">
        <p><em>Report generated by AI-Powered Dataset Analyzer | Powered by Google Gemini AI</em></p>
    </div>
</body>
</html>
"""
        
        return html_content
        
    except Exception as e:
        print(f"Error generating HTML report: {str(e)}")
        return f"""
<!DOCTYPE html>
<html>
<head><title>Error</title></head>
<body>
    <h1>Error Generating Report</h1>
    <p>Unable to generate report due to: {str(e)}</p>
</body>
</html>
"""

def generate_txt_report(data):
    """Generate TXT report with comprehensive analysis"""
    try:
        from datetime import datetime
        
        # Start building the report
        report = ""
        report += "=" * 60 + "\n"
        report += "           DATASET ANALYSIS REPORT\n"
        report += "=" * 60 + "\n"
        report += f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n"
        report += "=" * 60 + "\n\n"
        
        stats = data.get('statistics', {})
        insights = data.get('insights', '')
        charts = data.get('charts', [])
        preview = data.get('preview', [])
        columns = data.get('columns', [])
        
        # Dataset Overview
        if stats and stats.get('overview'):
            overview = stats['overview']
            report += "DATASET OVERVIEW\n"
            report += "-" * 40 + "\n"
            report += f"Total Rows: {overview.get('total_rows', 0):,}\n"
            report += f"Total Columns: {overview.get('total_columns', 0)}\n"
            report += f"Numerical Columns: {overview.get('numerical_columns', 0)}\n"
            report += f"Missing Values: {overview.get('total_missing_values', 0)}\n\n"
        
        # AI Insights
        if insights:
            report += "AI-GENERATED INSIGHTS\n"
            report += "-" * 40 + "\n"
            
            # Clean up insights
            cleaned_insights = insights
            if "AUTOMATED DATA INSIGHTS REPORT" in cleaned_insights:
                if "AI-Generated Analysis:" in cleaned_insights:
                    parts = cleaned_insights.split("AI-Generated Analysis:")
                    if len(parts) > 1:
                        cleaned_insights = "AI-Generated Analysis:" + parts[1]
                elif "Executive Summary" in cleaned_insights:
                    parts = cleaned_insights.split("Executive Summary")
                    if len(parts) > 1:
                        cleaned_insights = "Executive Summary" + parts[1]
            
            # Clean markdown formatting
            cleaned_insights = cleaned_insights.replace('# ', '').replace('## ', '').replace('### ', '')
            cleaned_insights = cleaned_insights.replace('**', '').replace('*', '').replace('---', '')
            
            report += cleaned_insights + "\n\n"
        
        # Statistical Analysis
        if stats:
            report += "STATISTICAL ANALYSIS\n"
            report += "-" * 40 + "\n"
            
            for column, values in stats.items():
                if column != 'overview' and isinstance(values, dict):
                    mean_val = values.get('mean')
                    median_val = values.get('median')
                    std_val = values.get('std')
                    min_val = values.get('min')
                    max_val = values.get('max')
                    
                    mean_str = f"{mean_val:.2f}" if mean_val is not None else 'N/A'
                    median_str = f"{median_val:.2f}" if median_val is not None else 'N/A'
                    std_str = f"{std_val:.2f}" if std_val is not None else 'N/A'
                    min_str = f"{min_val:.2f}" if min_val is not None else 'N/A'
                    max_str = f"{max_val:.2f}" if max_val is not None else 'N/A'
                    
                    report += f"\n{column}:\n"
                    report += f"  Mean: {mean_str}\n"
                    report += f"  Median: {median_str}\n"
                    report += f"  Std Dev: {std_str}\n"
                    report += f"  Min: {min_str}\n"
                    report += f"  Max: {max_str}\n"
                    report += f"  Missing: {values.get('missing', 0)}\n"
            
            report += "\n"
        
        # Visualizations
        if charts:
            report += "VISUALIZATIONS\n"
            report += "-" * 40 + "\n"
            report += f"Total Visualizations Generated: {len(charts)}\n\n"
            
            for i, chart in enumerate(charts, 1):
                chart_title = chart.get('title', f'Chart {i}')
                chart_type = chart.get('type', 'Unknown')
                report += f"{i}. {chart_title} ({chart_type})\n"
            
            report += "\n"
            
            # Add summary of visualizations
            chart_types = [chart.get('type', 'Unknown') for chart in charts]
            unique_types = list(set(chart_types))
            
            for chart_type in unique_types:
                count = chart_types.count(chart_type)
                if chart_type.lower() == 'bar':
                    report += f"• {count} Bar Chart(s): Display categorical data distribution\n"
                elif chart_type.lower() == 'line':
                    report += f"• {count} Line Chart(s): Show trends and time-series patterns\n"
                elif chart_type.lower() == 'pie':
                    report += f"• {count} Pie Chart(s): Illustrate proportional relationships\n"
                elif chart_type.lower() == 'histogram':
                    report += f"• {count} Histogram(s): Reveal data distribution patterns\n"
                elif chart_type.lower() == 'heatmap':
                    report += f"• {count} Heatmap(s): Show correlation between variables\n"
                else:
                    report += f"• {count} {chart_type} Chart(s): Interactive visualization\n"
            
            report += "\n"

        # Data Preview
        if preview and columns:
            report += "DATA PREVIEW (First 10 Rows)\n"
            report += "-" * 40 + "\n"
            report += '\t'.join(columns) + '\n'
            for row in preview:
                values = [str(row.get(col, '-')) for col in columns]
                report += '\t'.join(values) + '\n'
            report += '\n'

        # Footer
        report += "Report generated by AI-Powered Dataset Analyzer\n"
        report += "Powered by Google Gemini AI\n"

        return report

    except Exception as e:
        print(f"Error generating TXT report: {str(e)}")
        raise e

def generate_comprehensive_report(data):
    """Generate comprehensive report from analysis data"""
    try:
        stats = data.get('statistics', {})
        insights = data.get('insights', '')
        charts = data.get('charts', [])
        preview = data.get('preview', [])
        columns = data.get('columns', [])
        
        from datetime import datetime
        report = f"""DATASET ANALYSIS REPORT
Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
========================================

"""
        
        # Dataset Overview
        if stats and stats.get('overview'):
            overview = stats['overview']
            report += f"""DATASET OVERVIEW
================
Total Rows: {overview.get('total_rows', 0):,}
Total Columns: {overview.get('total_columns', 0)}
Numerical Columns: {overview.get('numerical_columns', 0)}
Categorical Columns: {overview.get('categorical_columns', 0)}
Missing Values: {overview.get('total_missing_values', 0)}

"""
        
        # AI Insights
        if insights:
            # Clean up the insights to remove the duplicate header section and markdown formatting
            cleaned_insights = insights
            
            # Remove the automated report header if present
            if "AUTOMATED DATA INSIGHTS REPORT" in cleaned_insights:
                # Split by the AI-Generated Analysis section
                if "AI-Generated Analysis:" in cleaned_insights:
                    parts = cleaned_insights.split("AI-Generated Analysis:")
                    if len(parts) > 1:
                        cleaned_insights = "AI-Generated Analysis:" + parts[1]
                else:
                    # If no AI-Generated Analysis section, look for Executive Summary
                    if "Executive Summary" in cleaned_insights:
                        parts = cleaned_insights.split("Executive Summary")
                        if len(parts) > 1:
                            cleaned_insights = "Executive Summary" + parts[1]
            
            # Clean up markdown formatting
            cleaned_insights = cleaned_insights.replace('# ', '').replace('## ', '').replace('### ', '')
            cleaned_insights = cleaned_insights.replace('**', '').replace('*', '')
            cleaned_insights = cleaned_insights.replace('---', '')
            
            # Clean up extra whitespace
            lines = cleaned_insights.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('---'):
                    cleaned_lines.append(line)
            
            cleaned_insights = '\n'.join(cleaned_lines)
            
            report += f"""AI-GENERATED INSIGHTS
=====================
{cleaned_insights}

"""
        
        # Statistical Analysis
        if stats:
            report += """STATISTICAL ANALYSIS
===================
"""
            
            # Overview stats
            if stats.get('overview'):
                overview = stats['overview']
                report += f"""Overview:
- Total Rows: {overview.get('total_rows', 0):,}
- Total Columns: {overview.get('total_columns', 0)}
- Numerical Columns: {overview.get('numerical_columns', 0)}
- Missing Values: {overview.get('total_missing_values', 0)}

"""
            
            # Column statistics
            for column, values in stats.items():
                if column != 'overview' and isinstance(values, dict):
                    # Format numeric values safely
                    mean_val = values.get('mean')
                    median_val = values.get('median')
                    std_val = values.get('std')
                    min_val = values.get('min')
                    max_val = values.get('max')
                    
                    mean_str = f"{mean_val:.2f}" if mean_val is not None else 'N/A'
                    median_str = f"{median_val:.2f}" if median_val is not None else 'N/A'
                    std_str = f"{std_val:.2f}" if std_val is not None else 'N/A'
                    min_str = f"{min_val:.2f}" if min_val is not None else 'N/A'
                    max_str = f"{max_val:.2f}" if max_val is not None else 'N/A'
                    
                    report += f"""{column}:
- Mean: {mean_str}
- Median: {median_str}
- Std Dev: {std_str}
- Min: {min_str}
- Max: {max_str}
- Missing: {values.get('missing', 0)}

"""
        
        # Data Preview
        if preview and columns:
            report += """DATA PREVIEW (First 10 Rows)
============================
"""
            
            # Headers
            report += '\t'.join(columns) + '\n'
            
            # Data rows
            for row in preview:
                values = [str(row.get(col, '-')) for col in columns]
                report += '\t'.join(values) + '\n'
            
            report += '\n'
        
        # Charts info
        if charts:
            report += """VISUALIZATIONS GENERATED
========================
"""
            for i, chart in enumerate(charts, 1):
                chart_title = chart.get('title', 'Untitled Chart')
                chart_type = chart.get('type', 'Unknown Type')
                
                report += f"{i}. {chart_title} ({chart_type})\n"
                
                # Add chart description based on type
                if chart_type.lower() == 'bar':
                    report += "   - Shows distribution of values across categories\n"
                elif chart_type.lower() == 'line':
                    report += "   - Displays trends and patterns over time/index\n"
                elif chart_type.lower() == 'pie':
                    report += "   - Illustrates proportional distribution of categories\n"
                elif chart_type.lower() == 'histogram':
                    report += "   - Shows frequency distribution of numerical data\n"
                elif chart_type.lower() == 'heatmap':
                    report += "   - Displays correlation matrix between variables\n"
                else:
                    report += "   - Interactive visualization for data exploration\n"
                
                report += "\n"
            
            report += f"Total Visualizations: {len(charts)}\n"
            report += "Note: Interactive charts are available in the web interface\n\n"
            
            # Add visualization insights
            report += """VISUALIZATION INSIGHTS
========================
"""
            chart_types = [chart.get('type', 'Unknown') for chart in charts]
            unique_types = list(set(chart_types))
            
            for chart_type in unique_types:
                count = chart_types.count(chart_type)
                if chart_type.lower() == 'bar':
                    report += f"• {count} Bar Chart(s): Display categorical data distribution\n"
                elif chart_type.lower() == 'line':
                    report += f"• {count} Line Chart(s): Show trends and time-series patterns\n"
                elif chart_type.lower() == 'pie':
                    report += f"• {count} Pie Chart(s): Illustrate proportional relationships\n"
                elif chart_type.lower() == 'histogram':
                    report += f"• {count} Histogram(s): Reveal data distribution patterns\n"
                elif chart_type.lower() == 'heatmap':
                    report += f"• {count} Heatmap(s): Show correlation between variables\n"
            
            report += "\nThese visualizations help identify:\n"
            report += "- Data distribution patterns\n"
            report += "- Trends and relationships\n"
            report += "- Outliers and anomalies\n"
            report += "- Statistical correlations\n\n"
        
        report += """Report generated by AI-Powered Dataset Analyzer
Powered by Google Gemini AI
"""
        
        return report
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        from datetime import datetime
        return f"""DATASET ANALYSIS REPORT
Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
========================================

ERROR: Unable to generate report due to: {str(e)}

Please try again or contact support.
"""

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
