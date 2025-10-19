# 📊 Automated Dataset Analyzer

An AI-powered web application for automated dataset analysis, statistical insights, and interactive visualizations. Upload your CSV or Excel files and get comprehensive analysis with AI-generated insights, statistical summaries, and beautiful visualizations.

## 🎯 Project Overview and Objectives

This application provides automated data analysis capabilities for researchers, data scientists, and business analysts who need quick insights from their datasets. The tool combines statistical analysis with AI-powered insights to deliver comprehensive reports that would typically require hours of manual work.

### Key Objectives:
- **Democratize Data Analysis**: Make advanced data analysis accessible to non-technical users
- **Accelerate Insights**: Provide instant analysis and visualizations for uploaded datasets
- **AI-Enhanced Reporting**: Generate intelligent insights and recommendations using Google Gemini AI
- **Interactive Experience**: Deliver results through an intuitive web interface with interactive charts

## 🚀 Features

### Core Functionality
- **File Upload Support**: CSV and Excel files (up to 200MB)
- **Automated Statistical Analysis**: Mean, median, mode, standard deviation, min/max values
- **AI-Powered Insights**: Comprehensive analysis using Google Gemini AI
- **Interactive Visualizations**: Bar charts, line charts, histograms, pie charts, correlation heatmaps
- **Data Quality Assessment**: Missing value analysis and data completeness metrics
- **Smart Sampling**: Automatic sampling for large datasets (>100K rows) to optimize performance

### Advanced Features
- **Real-time Processing**: Fast analysis with progress indicators
- **Comprehensive Reports**: Executive summaries, trend analysis, and actionable recommendations
- **Data Preview**: First 10 rows preview with proper formatting
- **Outlier Detection**: Statistical outlier identification using IQR method
- **Correlation Analysis**: Heatmap visualization of variable relationships

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Google Generative AI**: AI-powered insights generation

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Modern styling with responsive design
- **JavaScript**: Interactive client-side functionality
- **Jinja2**: Template engine for dynamic content

### Deployment
- **Gunicorn**: WSGI HTTP Server
- **Uvicorn**: ASGI server for FastAPI
- **Heroku**: Cloud deployment platform

### Dependencies
- **python-multipart**: File upload handling
- **openpyxl**: Excel file processing
- **python-dotenv**: Environment variable management

## 📦 Setup and Installation Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

### Local Development Setup

1. **Clone the repository:**
```bash
git clone https://github.com/MULTIMETA/dataset-analyzer.git
cd dataset-analyzer
```

2. **Create a virtual environment:**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

5. **Run the application:**
```bash
# Development server
python app/main.py

# Or using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

6. **Access the application:**
Open your browser and navigate to `http://localhost:8000`

### Production Setup

1. **Install production dependencies:**
```bash
pip install gunicorn
```

2. **Run with Gunicorn:**
```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 🔌 API Endpoints Documentation

### Core Endpoints

#### `GET /`
- **Description**: Main application interface
- **Response**: HTML page with upload form and analysis interface
- **Content-Type**: `text/html`

#### `POST /upload`
- **Description**: Upload and analyze dataset files
- **Request Body**: Multipart form data with file
- **Supported Formats**: CSV, XLS, XLSX
- **File Size Limit**: 200MB
- **Response**: JSON with analysis results
```json
{
  "statistics": {
    "overview": {
      "total_rows": 1000,
      "total_columns": 5,
      "numerical_columns": 3,
      "categorical_columns": 2,
      "total_missing_values": 0
    },
    "column_name": {
      "mean": 25.5,
      "median": 24.0,
      "std": 5.2,
      "min": 10.0,
      "max": 50.0,
      "missing": 0,
      "count": 1000
    }
  },
  "charts": [
    {
      "type": "bar",
      "title": "Bar Chart - Column Name",
      "data": "plotly_json_data"
    }
  ],
  "insights": "AI-generated analysis text...",
  "preview": [
    {"column1": "value1", "column2": "value2"}
  ],
  "columns": ["column1", "column2", "column3"]
}
```

#### `GET /health`
- **Description**: Health check endpoint
- **Response**: Application status
```json
{
  "status": "healthy",
  "message": "API is running",
  "ai_provider": "Google Gemini"
}
```

#### `GET /test-api-key`
- **Description**: Test Google Gemini API key configuration
- **Response**: API key status and preview
```json
{
  "status": "success",
  "message": "Gemini API key is loaded",
  "key_preview": "AIzaSyC...xyz1",
  "provider": "Google Gemini"
}
```

#### `GET /list-models`
- **Description**: List available Google Gemini models
- **Response**: Available AI models for analysis
```json
{
  "status": "success",
  "models": [
    {
      "name": "models/gemini-2.0-flash-exp",
      "supported_methods": ["generateContent"]
    }
  ]
}
```

### Error Responses

#### 400 Bad Request
```json
{
  "detail": "Unsupported format"
}
```

#### 413 Payload Too Large
```json
{
  "detail": "File exceeds 200MB"
}
```

#### 500 Internal Server Error
```json
{
  "detail": "Error message description"
}
```

## 🚀 Deployment Instructions

### Heroku Deployment

1. **Install Heroku CLI** and login:
```bash
heroku login
```

2. **Create Heroku app:**
```bash
heroku create your-app-name
```

3. **Set environment variables:**
```bash
heroku config:set GOOGLE_API_KEY=your_google_gemini_api_key
```

4. **Deploy:**
```bash
git push heroku main
```

5. **Open application:**
```bash
heroku open
```

### Docker Deployment

1. **Create Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

2. **Build and run:**
```bash
docker build -t dataset-analyzer .
docker run -p 8000:8000 -e GOOGLE_API_KEY=your_key dataset-analyzer
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google Gemini API key for AI insights | Yes |
| `PORT` | Server port (default: 8000) | No |
| `HOST` | Server host (default: 0.0.0.0) | No |

## ⚠️ Known Limitations and Future Improvements

### Current Limitations

1. **File Size Constraints**
   - Maximum file size: 200MB
   - Large datasets (>100K rows) are automatically sampled to 50K rows
   - Memory usage scales with dataset size

2. **Supported Formats**
   - Currently supports CSV, XLS, and XLSX files only
   - No support for JSON, Parquet, or other formats

3. **AI Analysis Limitations**
   - Requires Google Gemini API key
   - AI insights depend on API availability and rate limits
   - Analysis quality varies with data complexity

4. **Performance Considerations**
   - Processing time increases with dataset size
   - Complex visualizations may be slow for large datasets
   - No caching mechanism for repeated analyses

5. **Security Considerations**
   - No user authentication or data persistence
   - Uploaded files are processed in memory only
   - No data encryption or secure storage

### Planned Future Improvements

#### Short-term (Next 3 months)
- [ ] **Enhanced File Support**
  - Add support for JSON, Parquet, and SQLite files
  - Implement file format auto-detection
  - Add data validation and error handling

- [ ] **Performance Optimizations**
  - Implement Redis caching for analysis results
  - Add background job processing for large files
  - Optimize memory usage for large datasets

- [ ] **User Experience Improvements**
  - Add progress bars for long-running analyses
  - Implement drag-and-drop file upload
  - Add export functionality for reports (PDF, Word)

#### Medium-term (3-6 months)
- [ ] **Advanced Analytics**
  - Machine learning model integration
  - Time series analysis capabilities
  - Advanced statistical tests (t-tests, ANOVA, etc.)
  - Custom visualization types (scatter plots, box plots)

- [ ] **User Management**
  - User authentication and authorization
  - User-specific data storage and history
  - Collaborative analysis features

- [ ] **API Enhancements**
  - RESTful API with comprehensive documentation
  - Webhook support for analysis completion
  - Batch processing endpoints

#### Long-term (6+ months)
- [ ] **Enterprise Features**
  - Multi-tenant architecture
  - Advanced security and compliance
  - Integration with popular BI tools
  - Custom dashboard creation

- [ ] **AI/ML Enhancements**
  - Custom model training on user data
  - Automated feature engineering
  - Predictive analytics capabilities
  - Natural language query interface

- [ ] **Scalability Improvements**
  - Microservices architecture
  - Kubernetes deployment support
  - Auto-scaling capabilities
  - Global CDN integration

### Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting

### Support

For technical support or feature requests:
- Create an issue on GitHub
- Contact: [support@multimeta.com]
- Documentation: [https://github.com/MULTIMETA/dataset-analyzer]

---

**Built with ❤️ using FastAPI, Pandas, and Google Gemini AI**
