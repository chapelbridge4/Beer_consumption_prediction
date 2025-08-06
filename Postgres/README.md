# PostgreSQL Database Module

This module manages PostgreSQL database operations, including connections, table creation, and data management for the ETLP pipeline.

## 📁 Module Contents

### Core Files

#### `config.py`
- **Purpose**: Configuration management for PostgreSQL connections
- **Features**:
  - INI file parsing for database credentials
  - Environment-specific configuration loading
  - Secure credential management
- **Configuration File**: `postgres.ini` (not included in repository)

#### `connection_pos.py` 
- **Purpose**: Database connection management
- **Features**:
  - Connection pooling
  - Error handling and retry logic
  - Connection health monitoring
- **Dependencies**: Uses `psycopg2` for PostgreSQL connectivity

#### `db_postgres.py`
- **Purpose**: Database schema and table management
- **Features**:
  - Table creation scripts
  - Database view generation
  - Schema migration support
  - Data integrity constraints

## ⚙️ Configuration Setup

### Database Configuration File

Create a `postgres.ini` file in the project root with the following structure:

```ini
[PostgreSQL]
host=localhost
database=etlp_database
user=your_username
password=your_password
port=5432
```

### Environment Variables (Alternative)
You can also use environment variables for configuration:
- `POSTGRES_HOST`
- `POSTGRES_DB`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_PORT`

## 🚀 Usage

### Basic Connection
```python
from Postgres.connection_pos import connect

# Establish database connection
conn = connect()
if conn:
    print("Successfully connected to PostgreSQL")
    # Your database operations here
    conn.close()
```

### Table Creation
```python
from Postgres.db_postgres import create_tables, crea_viste

# Initialize database schema
create_tables()

# Create analytical views
crea_viste()
```

### Configuration Loading
```python
from Postgres.config import load_config

# Load database configuration
config = load_config()
print(f"Connecting to database: {config['database']}")
```

## 🗄️ Database Schema

The module creates and manages several key tables:

### Core Data Tables
- **Alcohol Consumption Data** - Global and regional consumption statistics
- **Country/Region Metadata** - Geographic and demographic information
- **Temporal Data** - Time-series data with proper indexing
- **Prediction Results** - Machine learning model outputs

### Views and Analytics
- **Missing Data Views** - Identify and track data gaps
- **Aggregated Statistics** - Pre-computed analytical views
- **Regional Summaries** - Geographic data aggregations

## 🔧 Functions Reference

### `load_config(filename, section)`
- **Parameters**: 
  - `filename` - Path to INI configuration file
  - `section` - Configuration section name (default: 'PostgreSQL')
- **Returns**: Dictionary of configuration parameters
- **Raises**: Exception if section not found

### `connect()`
- **Returns**: psycopg2 connection object
- **Features**: Auto-retry on connection failure
- **Error Handling**: Comprehensive error logging

### `create_tables()`
- **Purpose**: Initialize complete database schema
- **Features**: 
  - Idempotent table creation
  - Foreign key relationships
  - Proper indexing strategy

### `crea_viste()`
- **Purpose**: Create analytical database views
- **Features**:
  - Performance-optimized views
  - Data quality monitoring views
  - Missing value analysis

## 🔒 Security Features

- **Credential Management**: Secure INI file configuration
- **Connection Security**: SSL support for production environments
- **SQL Injection Prevention**: Parameterized queries throughout
- **Access Control**: Role-based database permissions

## 📊 Performance Optimization

- **Connection Pooling**: Efficient connection reuse
- **Index Strategy**: Optimized for analytical queries
- **Query Optimization**: Pre-computed views for common operations
- **Batch Operations**: Bulk data loading capabilities

## 🐛 Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check PostgreSQL service status
   - Verify host and port settings
   - Confirm firewall configurations

2. **Authentication Failed**
   - Verify username and password
   - Check PostgreSQL authentication settings
   - Ensure database exists

3. **Permission Denied**
   - Verify user has necessary privileges
   - Check database and table permissions
   - Review role assignments

### Debug Mode
Enable detailed logging by setting environment variable:
```bash
export POSTGRES_DEBUG=true
```

## 🔄 Integration with ETL Pipeline

This module integrates seamlessly with:
- **Data Extraction**: Stores processed API data
- **Data Transformation**: Provides staging tables
- **Machine Learning**: Stores model results and features
- **Analytics**: Serves processed data to visualization tools

## 📈 Monitoring and Maintenance

- **Connection Health**: Automatic connection testing
- **Query Performance**: Slow query logging
- **Data Quality**: Automated data validation views
- **Backup Integration**: Compatible with standard PostgreSQL backup tools

---

*This module provides robust PostgreSQL integration for the ETLP data pipeline.*