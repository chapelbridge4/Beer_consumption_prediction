# SQL Server Integration Module

This module should provide comprehensive SQL Server database integration, including connections, data migration, and cross-database operations for the ETLP pipeline. I removed those files for privacy.

## 📁 Module Components

### Connection Management

#### `connection.py`
- **Purpose**: SQL Server connection handling using SQLAlchemy
- **Features**:
  - Dynamic connection string generation
  - ODBC driver integration
  - Connection health monitoring
  - Error handling and retry logic
- **Configuration**: Uses `sql_server.ini` configuration file

#### `connessione_postgres_p.py`
- **Purpose**: PostgreSQL connection management within SQL Server module
- **Features**: Cross-database connectivity for data migration
- **Use Case**: Facilitates data transfer between SQL Server and PostgreSQL

### Core Operations

#### `sql.py`
- **Purpose**: Main SQL operations and data processing
- **Key Functions**:
  - `extract()` - Extract data from SQL Server tables
  - `load()` - Load processed data into PostgreSQL
  - `load_odata()` - Load external API data into PostgreSQL
  - `truncate_all_tables()` - Clean all PostgreSQL tables
  - `load_nazioni()` - Load country/nation reference data
  - `load_predizioni()` - Load machine learning predictions

## ⚙️ Configuration

### Database Configuration File

Create a `sql_server.ini` file in the project root:

```ini
[sql_server]
server=your_server_name
database=your_database_name  
user=your_username
password=your_password
driver=ODBC Driver 17 for SQL Server
```

### Connection String Format
The module automatically generates SQLAlchemy connection strings:
```
mssql+pyodbc://username:password@server/database?driver=ODBC+Driver+17+for+SQL+Server
```

## 🚀 Usage Examples

### Basic Connection
```python
from sql_server.connection import connect

# Connect to SQL Server
engine = connect()
if engine:
    print("Successfully connected to SQL Server")
    # Perform operations
```

### Data Operations
```python
from sql_server.sql import extract, load, load_odata, truncate_all_tables

# Clean existing data
truncate_all_tables()

# Extract data from SQL Server
extract()

# Load external API data
load_odata()

# Load processed data
load()
```

### Cross-Database Migration
```python
from sql_server.sql import extract, load

# Complete migration pipeline
extract()  # From SQL Server
load()     # To PostgreSQL
```

## 🔄 Data Pipeline Integration

### ETL Workflow
1. **Extract Phase**
   - Connect to SQL Server source
   - Extract structured data
   - Apply initial data validation

2. **Transform Phase** 
   - Data cleaning and normalization
   - Format standardization
   - Business logic application

3. **Load Phase**
   - Bulk insert into PostgreSQL
   - Index optimization
   - Data integrity verification

### Key Functions Overview

#### `extract()`
- **Purpose**: Extract data from SQL Server tables
- **Features**:
  - Incremental data loading
  - Data type preservation
  - Batch processing for large datasets
  - Error logging and recovery

#### `load()`
- **Purpose**: Load data from SQL Server into PostgreSQL
- **Features**:
  - Automatic schema mapping
  - Conflict resolution
  - Transaction management
  - Progress monitoring

#### `load_odata()`
- **Purpose**: Load external API data into PostgreSQL
- **Features**:
  - API data integration
  - Format harmonization
  - Duplicate detection
  - Data validation

#### `truncate_all_tables()`
- **Purpose**: Clean PostgreSQL tables for fresh data load
- **Features**:
  - Safe cascading truncation
  - Foreign key handling
  - Transaction rollback on errors
  - Confirmation prompts

#### `load_nazioni()`
- **Purpose**: Load country/nation reference data
- **Features**:
  - Geographic data normalization
  - ISO code mapping
  - Multi-language support
  - Historical country changes

#### `load_predizioni()`
- **Purpose**: Load machine learning prediction results
- **Features**:
  - Model result integration
  - Prediction metadata storage
  - Version control for models
  - Performance metrics tracking

## 🔧 Advanced Features

### Performance Optimization
- **Bulk Operations**: Optimized for large dataset transfers
- **Connection Pooling**: Efficient resource utilization
- **Parallel Processing**: Multi-threaded data operations
- **Memory Management**: Streaming for large datasets

### Data Quality Assurance
- **Validation Rules**: Comprehensive data validation
- **Error Logging**: Detailed operation logging
- **Rollback Mechanisms**: Safe operation execution
- **Data Profiling**: Automated data quality metrics

### Cross-Database Compatibility
- **Type Mapping**: Automatic SQL Server to PostgreSQL type conversion
- **Encoding Handling**: Character set compatibility
- **Date Format Standardization**: Consistent temporal data
- **Null Value Management**: Proper null handling across systems

## 🛠️ Dependencies

- **SQLAlchemy** - Database abstraction layer
- **pyodbc** - SQL Server connectivity
- **psycopg2** - PostgreSQL connectivity
- **pandas** - Data manipulation and analysis
- **configparser** - Configuration file management

## 🔒 Security Considerations

- **Credential Management**: Secure INI file storage
- **Connection Encryption**: SSL/TLS support
- **SQL Injection Prevention**: Parameterized queries
- **Access Logging**: Operation audit trails

## 📊 Monitoring and Maintenance

### Operation Monitoring
- **Progress Tracking**: Real-time operation status
- **Performance Metrics**: Query execution times  
- **Error Rates**: Operation success/failure tracking
- **Resource Usage**: Memory and CPU utilization

### Maintenance Tasks
- **Connection Health Checks**: Automatic connection testing
- **Data Consistency Verification**: Cross-database validation
- **Performance Tuning**: Query optimization suggestions
- **Log Rotation**: Automated log file management

## 🐛 Troubleshooting

### Common Issues

1. **ODBC Driver Not Found**
   ```bash
   # Install ODBC Driver 17 for SQL Server
   # Windows: Download from Microsoft
   # Linux: Use package manager
   sudo apt-get install msodbcsql17
   ```

2. **Connection Timeout**
   - Check network connectivity
   - Verify firewall settings
   - Increase connection timeout values

3. **Authentication Failures**
   - Verify SQL Server authentication mode
   - Check user permissions
   - Validate configuration file syntax

### Debug Mode
Enable detailed logging:
```bash
export SQL_SERVER_DEBUG=true
```

---

*This module enables seamless integration between SQL Server and PostgreSQL databases within the ETLP pipeline.*
