# Automated Dataset Analyzer

An AI-powered web application for automated dataset analysis, statistical insights, and interactive visualizations. Upload your CSV or Excel files and get comprehensive analysis with AI-generated insights, statistical summaries, and beautiful visualizations.

## Project Overview and Objectives

This application provides automated data analysis capabilities for researchers, data scientists, and business analysts who need quick insights from their datasets. The tool combines statistical analysis with AI-powered insights to deliver comprehensive reports that would typically require hours of manual work.

### Key Objectives:
- **Democratize Data Analysis**: Make advanced data analysis accessible to non-technical users
- **Accelerate Insights**: Provide instant analysis and visualizations for uploaded datasets
- **AI-Enhanced Reporting**: Generate intelligent insights and recommendations using Google Gemini AI
- **Interactive Experience**: Deliver results through an intuitive web interface with interactive charts

## Features

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

## Technology Stack

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

### Dependencies
- **python-multipart**: File upload handling
- **openpyxl**: Excel file processing
- **python-dotenv**: Environment variable management

##  Setup and Installation Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

### Local Development Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Abinaya-29751/StatCraft-.git
cd StatCraft
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

## API Endpoints Documentation

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
## Known Limitations and Future Improvements

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

#### Short-term 
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

#### Medium-term 
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

## ScreenShots

<img width="940" height="417" alt="image" src="https://github.com/user-attachments/assets/87c6227e-1015-43d9-a4ea-a3560e38a185" />
<img width="940" height="409" alt="image" src="https://github.com/user-attachments/assets/1db2aa0f-98cd-4cc5-b690-68b749f3e75e" />
<img width="940" height="438" alt="image" src="https://github.com/user-attachments/assets/620122f4-d738-40c3-b4ea-6d6a38c2ad10" />
<img width="940" height="376" alt="image" src="https://github.com/user-attachments/assets/04634909-66d8-446a-b0ac-490a3efe9ed5" />
<img width="940" height="430" alt="image" src="https://github.com/user-attachments/assets/9dcd7150-3963-4a7c-949a-82d26434fc61" />
<img width="940" height="416" alt="image" src="https://github.com/user-attachments/assets/6ae2e393-e5cf-4763-85f4-2bdd00f84f8f" />
<img width="940" height="448" alt="image" src="https://github.com/user-attachments/assets/3c8e3004-9d63-4c51-a7ed-ca45a15f96d5" />
<img width="940" height="191" alt="image" src="https://github.com/user-attachments/assets/822c35f0-3974-4446-8db3-576c4b867f92" />
<img width="940" height="379" alt="image" src="https://github.com/user-attachments/assets/cbf0aff8-7f3e-4697-96a4-a8689b4f13f4" />
<img width="940" height="348" alt="image" src="https://github.com/user-attachments/assets/f5540b0d-66ce-4ffc-9507-c2728797c2cc" />
<img width="940" height="404" alt="image" src="https://github.com/user-attachments/assets/b8b1b35e-0083-44ac-97f0-fc9ec8d221dc" />
<img width="940" height="399" alt="image" src="https://github.com/user-attachments/assets/9e9e0cec-5242-4513-80ca-19ca7a0b988f" />
<img width="940" height="468" alt="image" src="https://github.com/user-attachments/assets/02d389fe-0462-4bea-abb6-effab89ab69c" />

## Demo Video

[![Watch the demo](https://img.icons8.com/ios-filled/100/play-button-circled.png)](https://drive.google.com/file/d/1sx1IZzA5T421tTjMzCJnwryeF398Pkyc/view)

> 📱 Click the image above to watch the demo.

> This video demonstrates the key features of the application.











**Built with ❤️ using FastAPI, Pandas, and Google Gemini AI**
