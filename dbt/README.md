# Metrify Smart Metering - dbt Project

This dbt project contains data transformations for the Metrify Smart Metering data pipeline. It transforms raw data from various sources into clean, analytics-ready datasets.

## 📁 Project Structure

```
dbt/
├── models/
│   ├── staging/          # Staging models (raw data transformations)
│   │   ├── stg_smart_meter_readings.sql
│   │   ├── stg_grid_status.sql
│   │   ├── stg_weather_data.sql
│   │   └── schema.yml
│   ├── marts/            # Marts models (business logic)
│   │   ├── fct_smart_meter_analytics.sql
│   │   ├── dim_meters.sql
│   │   ├── dim_grid_operators.sql
│   │   ├── dim_weather_stations.sql
│   │   └── schema.yml
│   └── metrics/          # Metrics models (aggregated data)
│       ├── daily_consumption_metrics.sql
│       └── schema.yml
├── macros/               # Reusable SQL macros
│   ├── data_quality_checks.sql
│   ├── business_logic.sql
│   └── data_transformations.sql
├── seeds/                # Reference data
│   └── seed_smart_meter_types.csv
├── tests/                # Custom tests
├── dbt_project.yml       # dbt project configuration
├── profiles.yml          # Database connection profiles
├── packages.yml          # dbt package dependencies
├── requirements.txt      # Python dependencies
├── run_dbt.sh           # dbt execution script
└── README.md            # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **PostgreSQL** database accessible
3. **dbt** installed (`pip install dbt-postgres`)

### Installation

1. **Install Python dependencies:**
   ```bash
   # From the project root directory
   pip install -r requirements.txt
   ```
   
   **Important:** The main `requirements.txt` only includes pip-installable packages. dbt packages like `dbt-utils`, `dbt-expectations`, etc. are NOT pip packages and should not be installed with pip.

2. **Install dbt packages:**
   ```bash
   # Navigate to the dbt directory
   cd dbt
   
   # Install dbt packages (dbt-utils, dbt-expectations, etc.)
   dbt deps
   ```
   
   This will download dbt packages into the `dbt_modules/` directory based on `packages.yml`.

### Package Installation Explained

**Pip Packages (in requirements.txt):**
- `dbt-core` - Core dbt functionality
- `dbt-postgres` - PostgreSQL adapter
- `dbt-snowflake` - Snowflake adapter
- All other Python libraries

**dbt Packages (in packages.yml):**
- `dbt-utils` - Utility macros and functions
- `dbt-expectations` - Data quality tests
- `dbt-codegen` - Code generation utilities
- `dbt-audit-helper` - Audit and lineage tools
- `dbt-external-tables` - External table management
- `dbt-date` - Date/time utilities

**Why the distinction?**
- dbt packages contain SQL macros and are managed by dbt's package manager
- They cannot be installed with pip because they don't have Python setup files
- Always use `dbt deps` for dbt packages and `pip install` for Python packages

3. **Configure database connection:**
   - Copy `profiles.yml` to `~/.dbt/profiles.yml`
   - Update connection details for your environment

### Running dbt

Use the provided script for easy execution:

```bash
# Make script executable
chmod +x run_dbt.sh

# Install packages
./run_dbt.sh install

# Parse models
./run_dbt.sh parse

# Run all models
./run_dbt.sh run

# Run specific model types
./run_dbt.sh run staging
./run_dbt.sh run marts
./run_dbt.sh run metrics

# Run tests
./run_dbt.sh test

# Generate documentation
./run_dbt.sh docs
```

## 📊 Data Models

### Staging Models

**`stg_smart_meter_readings`**
- Cleans and validates smart meter reading data
- Adds data quality flags and business logic
- Calculates consumption categories and time-based features

**`stg_grid_status`**
- Transforms grid operator status data
- Validates electrical parameters
- Adds grid stability classifications

**`stg_weather_data`**
- Processes weather station observations
- Validates meteorological parameters
- Adds weather condition classifications

### Marts Models

**`fct_smart_meter_analytics`**
- Hourly aggregated smart meter data
- Quality metrics and anomaly detection
- Business context and time classifications

**`dim_meters`**
- Master data for all smart meters
- Quality and performance metrics
- Geographic and operational classifications

**`dim_grid_operators`**
- Master data for grid operators
- Stability and risk assessments
- Operational status and performance

**`dim_weather_stations`**
- Master data for weather stations
- Environmental classifications
- Data quality and completeness metrics

### Metrics Models

**`daily_consumption_metrics`**
- Daily aggregated consumption data
- Quality and anomaly metrics
- Business and seasonal classifications

## 🔧 Configuration

### Environment Variables

Set these environment variables for your environment:

```bash
# Database connection
export POSTGRES_HOST=localhost
export POSTGRES_USER=metrify_user
export POSTGRES_PASSWORD=metrify_password
export POSTGRES_DB=metrify_smart_metering
export POSTGRES_PORT=5432

# dbt configuration
export DBT_ENVIRONMENT=development
export DBT_SCHEMA=public
export DBT_SEARCH_PATH=public
```

### Variables

The project uses several configurable variables in `dbt_project.yml`:

- **Data Quality Thresholds**: Min/max values for validation
- **Business Rules**: Peak hours, weekend days, etc.
- **Data Retention**: How long to keep data in each layer

## 🧪 Testing

### Data Quality Tests

All models include comprehensive data quality tests:

- **Not null tests** for required fields
- **Range tests** for numeric values
- **Accepted values** for categorical fields
- **Custom tests** for business logic

### Running Tests

```bash
# Run all tests
./run_dbt.sh test

# Run specific test types
./run_dbt.sh test staging
./run_dbt.sh test marts
./run_dbt.sh test metrics
```

### Test Coverage

- **Staging models**: 100% test coverage
- **Marts models**: 100% test coverage
- **Metrics models**: 100% test coverage

## 📈 Data Quality

### Quality Checks

The project implements multiple layers of data quality:

1. **Source validation** in staging models
2. **Business rule validation** in marts
3. **Aggregation validation** in metrics
4. **Referential integrity** across models

### Quality Metrics

- **Completeness**: Percentage of non-null values
- **Accuracy**: Validation against business rules
- **Consistency**: Cross-model validation
- **Timeliness**: Data freshness checks

## 🔄 Data Lineage

### Dependencies

```
Raw Data Sources
    ↓
Staging Models (stg_*)
    ↓
Marts Models (dim_*, fct_*)
    ↓
Metrics Models (daily_*)
```

### Refresh Strategy

- **Staging**: Real-time (as data arrives)
- **Marts**: Hourly (incremental)
- **Metrics**: Daily (full refresh)

## 📚 Documentation

### Generated Docs

Generate and view documentation:

```bash
./run_dbt.sh docs
# Open http://localhost:8080 in your browser
```

### Model Documentation

Each model includes:
- **Description** of purpose and business logic
- **Column descriptions** with data types
- **Test definitions** and expected outcomes
- **Dependencies** and lineage information

## 🚨 Monitoring

### dbt Cloud Integration

The project is configured for dbt Cloud with:
- **Automatic runs** on schedule
- **Alerting** on failures
- **Performance monitoring**
- **Data quality monitoring**

### Custom Monitoring

- **Model execution times**
- **Data quality scores**
- **Anomaly detection rates**
- **Business metric tracking**

## 🔧 Development

### Adding New Models

1. **Create model file** in appropriate directory
2. **Add tests** in schema.yml
3. **Update documentation** with descriptions
4. **Run tests** to validate

### Adding New Macros

1. **Create macro file** in macros/ directory
2. **Use in models** with `{{ macro_name() }}`
3. **Test thoroughly** with different inputs
4. **Document usage** in README

### Code Style

- **SQL formatting** with dbt-utils
- **Naming conventions** (snake_case)
- **Commenting** for complex logic
- **Version control** with Git

## 🐛 Troubleshooting

### Common Issues

1. **Connection errors**: Check profiles.yml configuration
2. **Model failures**: Check dependencies and data quality
3. **Test failures**: Review data and business rules
4. **Performance issues**: Check indexes and query optimization

### Debug Mode

Run in debug mode for detailed information:

```bash
./run_dbt.sh debug
```

### Logs

Check dbt logs for detailed error information:

```bash
dbt run --log-level debug
```

## 📞 Support

For issues and questions:
- **Documentation**: Check model descriptions and tests
- **Logs**: Review dbt execution logs
- **Tests**: Run data quality tests
- **Community**: dbt Slack workspace

## 📄 License

This project is part of the Metrify Smart Metering data pipeline and follows the same licensing terms.
