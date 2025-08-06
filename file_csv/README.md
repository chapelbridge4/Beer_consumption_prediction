# CSV Data Files - Raw Dataset Repository

This directory contains raw datasets in CSV and Excel formats from various international statistical organizations, serving as the foundation for the ETLP pipeline's data analysis.

## 📊 Dataset Overview

### Data Sources

- **WHO** - World Health Organization (https://www.who.int/)
- **ISTAT** - Italian National Institute of Statistics (https://www.istat.it/)
- **Eurostat** - European Statistical Office

## 📁 File Descriptions

### Core Datasets

#### `dati_global.csv`
- **Source**: WHO Global Health Observatory
- **Content**: Global alcohol consumption statistics
- **Unit of Measure**: Alcohol recorded per capita (15+) consumption in liters of pure alcohol
- **Coverage**: Worldwide data by country and year
- **Time Period**: Multi-year historical data

#### `dati_istat_regioni.csv` 
- **Source**: ISTAT Regional Statistics
- **Content**: Italian regional alcohol consumption patterns
- **Unit of Measure**: Per 100 people with same characteristics, values in thousands (number of people consuming alcohol)
- **Coverage**: All Italian regions
- **Demographics**: Age groups, gender, socioeconomic factors

#### `dati_ita.csv`
- **Source**: ISTAT National Statistics
- **Content**: Italian national alcohol consumption data
- **Unit of Measure**: Per 100 people with same characteristics, values in thousands (number of people consuming alcohol)
- **Coverage**: National-level aggregated data
- **Breakdown**: By demographic characteristics and consumption patterns

#### `ds-056120__custom_3278652_spreadsheet.xlsx`
- **Source**: Custom industry dataset
- **Content**: Beer consumption and economic data
- **Unit of Measure**: 
  - Volume: Liters of beer
  - Value: Euro currency
- **Purpose**: Economic analysis and market correlation studies

## 📈 Data Characteristics

### Global Data (`dati_global.csv`)
- **Primary Metric**: Liters of pure alcohol per capita
- **Population**: Adults 15+ years
- **Recording Type**: Officially recorded consumption
- **Geographic Scope**: 194+ countries
- **Temporal Resolution**: Annual data points

### Italian Regional Data (`dati_istat_regioni.csv`) 
- **Primary Metric**: Consumption prevalence rates
- **Population Base**: Per 100 people with same characteristics
- **Geographic Granularity**: Regional (20 regions)
- **Demographic Segmentation**: 
  - Age groups (15-24, 25-44, 45-64, 65+)
  - Gender (Male, Female)
  - Education levels
  - Employment status

### Italian National Data (`dati_ita.csv`)
- **Aggregation Level**: National totals and averages
- **Time Series**: Multi-year trend data
- **Comparison Metrics**: Inter-regional comparisons
- **Policy Indicators**: Public health monitoring metrics

### Economic Data (`ds-056120__custom_3278652_spreadsheet.xlsx`)
- **Market Data**: Beer industry statistics
- **Financial Metrics**: Revenue and volume correlations
- **Economic Indicators**: Price elasticity and market trends
- **Business Intelligence**: Commercial consumption patterns

## 🔄 Data Processing Pipeline

### Input Stage
1. **Raw Data Ingestion** - Files loaded as-is from source organizations
2. **Format Validation** - File integrity and structure verification
3. **Encoding Standardization** - UTF-8 conversion for international compatibility

### Transformation Stage
1. **Data Cleaning** - Missing value identification and handling
2. **Normalization** - Unit standardization across datasets
3. **Harmonization** - Geographic and temporal alignment
4. **Quality Assessment** - Statistical validation and outlier detection

### Output Stage
1. **Database Loading** - Integration with PostgreSQL tables
2. **Feature Engineering** - Derived metrics and calculated fields
3. **ML Preparation** - Dataset preparation for prediction models

## 🌍 Geographic Coverage

### Global Dataset
- **Scope**: Worldwide coverage (194+ countries)
- **Regions**: All WHO regions included
- **Standards**: ISO country codes and UN geographic classifications

### Italian Focus
- **National Level**: Complete Italy coverage
- **Regional Level**: All 20 Italian administrative regions
- **Standards**: ISTAT territorial classification (NUTS-2)

## ⏰ Temporal Coverage

- **Historical Range**: Multi-year time series (varies by dataset)
- **Update Frequency**: Annual updates from source organizations
- **Data Lag**: Typically 1-2 years behind current year
- **Trend Analysis**: Sufficient historical depth for trend analysis

## 📊 Data Quality Features

### Completeness
- **Coverage Assessment**: Geographic and temporal completeness tracking
- **Missing Data Patterns**: Systematic gap identification
- **Imputation Readiness**: Prepared for statistical imputation methods

### Accuracy
- **Source Validation**: Cross-reference with original publications
- **Consistency Checks**: Internal logical consistency verification
- **Outlier Detection**: Statistical outlier identification and flagging

### Reliability  
- **Methodology Documentation**: Source methodology preservation
- **Update Tracking**: Version control and change documentation
- **Audit Trail**: Complete data lineage tracking

## 🛠️ Usage in ML Pipeline

### Feature Engineering
- **One-Hot Encoding**: Country categorical variables prepared
- **Temporal Features**: Year-over-year change calculations
- **Geographic Features**: Regional and country-level indicators
- **Economic Features**: GDP and market correlation variables

### Model Training
- **Target Variables**: Consumption levels and patterns
- **Predictor Variables**: Demographic, economic, and temporal features
- **Cross-Validation**: Geographic and temporal split strategies
- **Evaluation Metrics**: Region-specific model performance

## 📋 Data Dictionary

### Common Fields
- **Country/Region**: Geographic identifier (ISO codes)
- **Year**: Temporal identifier (YYYY format)
- **Consumption_Metric**: Primary measurement value
- **Unit**: Measurement unit specification
- **Source_Method**: Data collection methodology
- **Quality_Flag**: Data quality indicator

### Demographic Dimensions
- **Age_Group**: Standardized age brackets
- **Gender**: Male/Female/Total categories  
- **Education**: Educational attainment levels
- **Urban_Rural**: Settlement type classification

## 🔍 Research Applications

- **Public Health Analysis**: Population health trend monitoring
- **Policy Evaluation**: Intervention impact assessment
- **Economic Research**: Market analysis and forecasting
- **Social Studies**: Cultural and behavioral pattern analysis
- **Predictive Modeling**: Consumption forecasting and risk assessment

---

*These datasets form the empirical foundation for comprehensive alcohol consumption analysis and predictive modeling in the ETLP pipeline.*