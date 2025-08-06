# ETLP - ETL Pipeline for Alcohol Consumption Data Analysis

A comprehensive ETL (Extract, Transform, Load) pipeline designed for analyzing global alcohol consumption data using machine learning predictions and data visualization.

## 🌟 Features

- **Multi-source Data Extraction**: Automated data collection from WHO, ISTAT, and Eurostat APIs
- **Database Integration**: Support for both PostgreSQL and SQL Server databases
- **Machine Learning Predictions**: Alcohol consumption forecasting models
- **Data Visualization**: Interactive charts and analytics
- **REST API**: FastAPI-based endpoints for data access
- **Data Processing**: Comprehensive ETL pipeline with data cleaning and transformation

## 🏗️ Project Structure

### Core Modules

- **`api_odata/`** - OData API integrations and data extraction scripts
- **`file_csv/`** - Raw CSV files and Excel spreadsheets containing base datasets
- **`Postgres/`** - PostgreSQL database scripts, connections, and table management
- **`prediction/`** - Machine learning models and prediction algorithms
- **`sql_server/`** - SQL Server integration and data migration scripts

### Key Files

- **`main.py`** - Main execution script and pipeline orchestrator
- **`requirements.txt`** - Python dependencies

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- SQL Server (optional)
- Required Python packages (see requirements.txt)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ETLP_PUBLIC
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Database Configuration**
   - Create configuration files for your databases:
     - `postgres.ini` for PostgreSQL settings
     - `sql_server.ini` for SQL Server settings (if used)
   
   Example `postgres.ini`:
   ```ini
   [PostgreSQL]
   host=localhost
   database=your_database
   user=your_username
   password=your_password
   port=5432
   ```

4. **Initialize Database**
   ```bash
   python main.py
   ```

## 📊 Data Sources

- **WHO (World Health Organization)** - Global alcohol consumption statistics
- **ISTAT** - Italian National Institute of Statistics
- **Eurostat** - European statistical data

## 🔧 Usage

### Running the ETL Pipeline

The main pipeline can be executed by running:

```bash
python main.py
```

### Available Operations

- `create_tables()` - Initialize database tables
- `extract()` - Extract data from SQL Server
- `load_odata()` - Load data from OData APIs
- `load_nazioni()` - Load country/nation data
- `load_predizioni()` - Load prediction results

## 🤖 Machine Learning Models

The project includes several pre-trained models for alcohol consumption prediction:

- `model_acquisti.pkl` - Purchase prediction model
- `model_consumi.pkl` - Consumption prediction model
- `modello_regIta.pkl` - Italian regional model
- Various specialized regional and zone models

## 🛠️ Technology Stack

- **Python 3.8+**
- **FastAPI** - REST API framework
- **PostgreSQL** - Primary database
- **SQL Server** - Secondary database support
- **SQLAlchemy** - Database ORM
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning
- **Matplotlib/Seaborn** - Data visualization
- **Requests** - HTTP client for API calls

## 📈 Data Pipeline Flow

1. **Extract** - Fetch data from multiple sources (WHO, ISTAT, Eurostat)
2. **Transform** - Clean, normalize, and process raw data
3. **Load** - Store processed data in PostgreSQL/SQL Server
4. **Analyze** - Generate insights and predictions
5. **Visualize** - Create charts and dashboards

## ⚡ Quick Start Example

```python
from main import main

# Run the complete ETL pipeline
main()
```

---

*This project demonstrates ETL pipeline development, data analytics, and machine learning implementation for alcohol consumption analysis.*
