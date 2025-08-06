# API OData - Data Extraction Module

This module handles data extraction and processing from various OData APIs, focusing on alcohol consumption statistics from international organizations.

## 📁 Files Overview

### Core Scripts

#### `api_global.py`
- **Purpose**: Direct data extraction from global statistical APIs
- **Output**: Saves extracted data to `output.json`
- **Data Sources**: WHO and other international health organizations

#### `api_istat.py` 
- **Purpose**: Data extraction from ISTAT (Italian National Institute of Statistics)
- **Source URL**: [ISTAT Alcohol Consumption Data](https://www.istat.it/it/archivio/244222)
- **Output**: `tavole_consumo_di_alcol_2019.xls`

#### `api_eurostats.py`
- **Purpose**: European statistical data extraction from Eurostat APIs
- **Focus**: EU alcohol consumption and related statistics

### Data Processing Scripts

#### `dati_global.py`
- **Input**: Pre-saved CSV file from `file_csv/dati_global.csv`
- **Process**: Data transformation, cleaning, and filtering
- **Output**: Processed `df_melt` DataFrame
- **Features**:
  - Data normalization
  - Missing value handling
  - Format standardization

#### `dati_istat.py`
- **Input**: Pre-saved CSV file from `file_csv/dati_ita.csv`
- **Process**: Italian-specific data transformation
- **Output**: Processed `df_ita` DataFrame
- **Focus**: Italian regional alcohol consumption patterns

#### `dati_istat_regioni.py`
- **Purpose**: Regional-level data processing for Italian territories
- **Features**: Geographic data aggregation and analysis

#### `dati_nazioni.py`
- **Purpose**: Country-level data harmonization and processing
- **Output**: Standardized national statistics

## 🔄 Data Flow

1. **Raw Data Extraction** → API calls to external sources
2. **Initial Processing** → Data cleaning and validation
3. **Transformation** → Format standardization and normalization
4. **Storage** → Save to CSV/JSON for further pipeline processing

## 🌍 Data Sources

- **WHO (World Health Organization)** - Global health statistics
- **ISTAT** - Italian national and regional statistics
- **Eurostat** - European Union statistical data

## 📊 Data Types

- **Alcohol Consumption**: Per capita consumption in liters of pure alcohol
- **Demographic Data**: Population statistics and regional breakdowns
- **Economic Indicators**: Related economic and social metrics
- **Temporal Data**: Multi-year trends and historical data

## 🚀 Usage

### Basic Data Extraction
```python
from api_odata import api_global, api_istat

# Extract global data
api_global.extract_global_data()

# Extract ISTAT data
api_istat.fetch_italian_data()
```

### Data Processing
```python
from api_odata import dati_global, dati_istat

# Process global dataset
global_df = dati_global.process_data()

# Process Italian dataset  
italian_df = dati_istat.process_italian_data()
```

## ⚙️ Configuration

Ensure proper API credentials and endpoints are configured before running extraction scripts. Some APIs may require authentication tokens or specific request headers.

## 📝 Output Formats

- **JSON**: Structured data from direct API calls
- **CSV**: Processed tabular data ready for analysis
- **Excel**: Formatted reports with multiple sheets
- **DataFrames**: In-memory Pandas objects for immediate use

## 🔧 Dependencies

- `requests` - HTTP API calls
- `pandas` - Data manipulation
- `openpyxl` - Excel file handling
- `json` - JSON data processing

## 📈 Data Quality

All scripts include data validation and quality checks:
- Missing value identification
- Data type validation
- Range and consistency checks
- Duplicate detection and handling

---

*This module serves as the primary data ingestion layer for the ETLP pipeline.*