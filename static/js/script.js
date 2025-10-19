// Show selected file info
document.getElementById('fileInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const fileSize = (file.size / (1024 * 1024)).toFixed(2);
        let infoText = `Selected: ${file.name} (${fileSize} MB)`;
        if (file.size > 50 * 1024 * 1024) {
            infoText += ' ⚠️ Large file';
            document.getElementById('fileInfo').style.color = 'orange';
        } else {
            document.getElementById('fileInfo').style.color = '#666';
        }
        document.getElementById('fileInfo').textContent = infoText;
    }
});

async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('⚠️ Please select a file first!');
        return;
    }
    
    const validExtensions = ['csv', 'xlsx', 'xls'];
    const fileExtension = file.name.split('.').pop().toLowerCase();
    
    if (!validExtensions.includes(fileExtension)) {
        alert('⚠️ Please upload a CSV or Excel file!');
        return;
    }
    
    if (file.size > 200 * 1024 * 1024) {
        alert(`⚠️ File too large. Max 200MB.`);
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    const loadingDiv = document.getElementById('loading');
    loadingDiv.innerHTML = `
        <div class="spinner"></div>
        <p>Processing dataset...</p>
    `;
    loadingDiv.style.display = 'block';
    document.getElementById('results').style.display = 'none';
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        alert('❌ Error: ' + error.message);
    } finally {
        loadingDiv.style.display = 'none';
    }
}

function displayResults(data) {
    document.getElementById('results').style.display = 'block';
    displayInsights(data.insights);
    displayStatistics(data.statistics);
    displayCharts(data.charts);
    displayPreview(data.preview, data.columns);
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

function displayInsights(insights) {
    console.log('Rendering insights with marked.js...');
    
    // Check if marked.js is loaded
    if (typeof marked === 'undefined') {
        console.error('Marked.js not loaded! Using fallback...');
        document.getElementById('insightsContent').innerHTML = 
            `<div class="insights-rendered"><pre>${insights}</pre></div>`;
        return;
    }
    
    // Configure marked.js
    marked.setOptions({
        breaks: true,
        gfm: true
    });
    
    // Convert markdown to HTML
    const htmlContent = marked.parse(insights);
    
    // Display with custom styling
    document.getElementById('insightsContent').innerHTML = 
        `<div class="insights-rendered">${htmlContent}</div>`;
    
    console.log(' Insights rendered successfully!');
}

function displayStatistics(stats) {
    const overview = stats.overview;
    const overviewHTML = `
        <div class="stat-card">
            <h3>${overview.total_rows.toLocaleString()}</h3>
            <p>Total Rows</p>
        </div>
        <div class="stat-card">
            <h3>${overview.total_columns}</h3>
            <p>Total Columns</p>
        </div>
        <div class="stat-card">
            <h3>${overview.numerical_columns}</h3>
            <p>Numerical Columns</p>
        </div>
        <div class="stat-card">
            <h3>${overview.total_missing_values}</h3>
            <p>Missing Values</p>
        </div>
    `;
    document.getElementById('statsOverview').innerHTML = overviewHTML;
    
    let detailsHTML = '<table><thead><tr><th>Column</th><th>Mean</th><th>Median</th><th>Std Dev</th><th>Min</th><th>Max</th><th>Missing</th></tr></thead><tbody>';
    
    for (const [column, values] of Object.entries(stats)) {
        if (column !== 'overview') {
            detailsHTML += `<tr>
                <td><strong>${column}</strong></td>
                <td>${values.mean?.toFixed(2) || 'N/A'}</td>
                <td>${values.median?.toFixed(2) || 'N/A'}</td>
                <td>${values.std?.toFixed(2) || 'N/A'}</td>
                <td>${values.min?.toFixed(2) || 'N/A'}</td>
                <td>${values.max?.toFixed(2) || 'N/A'}</td>
                <td>${values.missing}</td>
            </tr>`;
        }
    }
    
    detailsHTML += '</tbody></table>';
    document.getElementById('statsDetails').innerHTML = detailsHTML;
}

function displayCharts(charts) {
    const container = document.getElementById('chartsContainer');
    container.innerHTML = '';
    
    if (charts.length === 0) {
        container.innerHTML = '<p>No visualizations available.</p>';
        return;
    }
    
    charts.forEach((chart, index) => {
        const chartDiv = document.createElement('div');
        chartDiv.className = 'chart-container';
        chartDiv.id = `chart${index}`;
        container.appendChild(chartDiv);
        
        const plotData = JSON.parse(chart.data);
        Plotly.newPlot(`chart${index}`, plotData.data, plotData.layout, { responsive: true });
    });
}

// function displayPreview(data, columns) {
//     if (!data || data.length === 0) {
//         document.getElementById('previewContent').innerHTML = '<p>No preview available.</p>';
//         return;
//     }
    
//     let html = '<table><thead><tr>';
//     columns.forEach(col => html += `<th>${col}</th>`);
//     html += '</tr></thead><tbody>';
    
//     data.forEach(row => {
//         html += '<tr>';
//         columns.forEach(col => {
//             const value = row[col];
//             html += `<td>${value !== null && value !== undefined ? value : '-'}</td>`;
//         });
//         html += '</tr>';
//     });
    
//     html += '</tbody></table>';
//     document.getElementById('previewContent').innerHTML = html;
// }

// console.log(' Script.js v10 loaded - Marked.js integration active');


// Display data preview
function displayPreview(data, columns) {
    if (!data || data.length === 0) {
        document.getElementById('previewContent').innerHTML = '<p>No preview available.</p>';
        return;
    }
    
    let html = '<table><thead><tr>';
    columns.forEach(col => html += `<th>${col}</th>`);
    html += '</tr></thead><tbody>';
    
    data.forEach(row => {
        html += '<tr>';
        columns.forEach(col => {
            const value = row[col];
            html += `<td>${value !== null && value !== undefined ? value : '-'}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    document.getElementById('previewContent').innerHTML = html;
}

console.log('✅ Dataset Analyzer ready');
