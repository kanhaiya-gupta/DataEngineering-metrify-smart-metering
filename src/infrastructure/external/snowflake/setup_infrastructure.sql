-- =============================================================================
-- METRIFY SMART METERING SNOWFLAKE INFRASTRUCTURE SETUP
-- Account: NEIVWLJ-BF21045
-- User: KGUPTA (ACCOUNTADMIN)
-- =============================================================================

-- Create Warehouses
-- =============================================================================
CREATE WAREHOUSE IF NOT EXISTS DEV_WH 
WITH 
    WAREHOUSE_SIZE = 'X-SMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    MIN_CLUSTER_COUNT = 1
    MAX_CLUSTER_COUNT = 1
    SCALING_POLICY = 'STANDARD'
    COMMENT = 'Development warehouse for Metrify smart metering system';

CREATE WAREHOUSE IF NOT EXISTS STAGING_WH 
WITH 
    WAREHOUSE_SIZE = 'SMALL'
    AUTO_SUSPEND = 300
    AUTO_RESUME = TRUE
    MIN_CLUSTER_COUNT = 1
    MAX_CLUSTER_COUNT = 3
    SCALING_POLICY = 'STANDARD'
    COMMENT = 'Staging warehouse for Metrify smart metering system';

CREATE WAREHOUSE IF NOT EXISTS PROD_WH 
WITH 
    WAREHOUSE_SIZE = 'LARGE'
    AUTO_SUSPEND = 600
    AUTO_RESUME = TRUE
    MIN_CLUSTER_COUNT = 2
    MAX_CLUSTER_COUNT = 10
    SCALING_POLICY = 'ECONOMY'
    COMMENT = 'Production warehouse for Metrify smart metering system';

CREATE WAREHOUSE IF NOT EXISTS ANALYTICS_WH 
WITH 
    WAREHOUSE_SIZE = 'X-LARGE'
    AUTO_SUSPEND = 1800
    AUTO_RESUME = TRUE
    MIN_CLUSTER_COUNT = 1
    MAX_CLUSTER_COUNT = 5
    SCALING_POLICY = 'STANDARD'
    COMMENT = 'Analytics warehouse for Metrify smart metering system';

-- Create Databases
-- =============================================================================
CREATE DATABASE IF NOT EXISTS METRIFY_OPERATIONAL
COMMENT = 'Operational data for smart metering system';

CREATE DATABASE IF NOT EXISTS METRIFY_ANALYTICS
COMMENT = 'Analytics and reporting data for smart metering system';

CREATE DATABASE IF NOT EXISTS METRIFY_ARCHIVE
COMMENT = 'Long-term archived data for smart metering system';

-- Create Schemas in METRIFY_OPERATIONAL
-- =============================================================================
USE DATABASE METRIFY_OPERATIONAL;

CREATE SCHEMA IF NOT EXISTS RAW
COMMENT = 'Raw data from external sources'
DATA_RETENTION_TIME_IN_DAYS = 7;

CREATE SCHEMA IF NOT EXISTS STAGING
COMMENT = 'Cleaned and validated data'
DATA_RETENTION_TIME_IN_DAYS = 30;

CREATE SCHEMA IF NOT EXISTS MARTS
COMMENT = 'Business-ready data marts'
DATA_RETENTION_TIME_IN_DAYS = 90;

CREATE SCHEMA IF NOT EXISTS ANALYTICS
COMMENT = 'Analytics and ML features'
DATA_RETENTION_TIME_IN_DAYS = 365;

-- Create Schemas in METRIFY_ANALYTICS
-- =============================================================================
USE DATABASE METRIFY_ANALYTICS;

CREATE SCHEMA IF NOT EXISTS RAW
COMMENT = 'Raw data from external sources'
DATA_RETENTION_TIME_IN_DAYS = 7;

CREATE SCHEMA IF NOT EXISTS STAGING
COMMENT = 'Cleaned and validated data'
DATA_RETENTION_TIME_IN_DAYS = 30;

CREATE SCHEMA IF NOT EXISTS MARTS
COMMENT = 'Business-ready data marts'
DATA_RETENTION_TIME_IN_DAYS = 90;

CREATE SCHEMA IF NOT EXISTS ANALYTICS
COMMENT = 'Analytics and ML features'
DATA_RETENTION_TIME_IN_DAYS = 365;

-- Create Schemas in METRIFY_ARCHIVE
-- =============================================================================
USE DATABASE METRIFY_ARCHIVE;

CREATE SCHEMA IF NOT EXISTS RAW
COMMENT = 'Raw data from external sources'
DATA_RETENTION_TIME_IN_DAYS = 7;

CREATE SCHEMA IF NOT EXISTS STAGING
COMMENT = 'Cleaned and validated data'
DATA_RETENTION_TIME_IN_DAYS = 30;

CREATE SCHEMA IF NOT EXISTS MARTS
COMMENT = 'Business-ready data marts'
DATA_RETENTION_TIME_IN_DAYS = 90;

CREATE SCHEMA IF NOT EXISTS ANALYTICS
COMMENT = 'Analytics and ML features'
DATA_RETENTION_TIME_IN_DAYS = 365;

-- Create Roles
-- =============================================================================
CREATE ROLE IF NOT EXISTS ANALYTICS_ROLE
COMMENT = 'Role for analytics and data processing';

CREATE ROLE IF NOT EXISTS PROD_ROLE
COMMENT = 'Role for production operations';

CREATE ROLE IF NOT EXISTS DEV_ROLE
COMMENT = 'Role for development operations';

-- Grant Warehouse Permissions to KGUPTA (ACCOUNTADMIN already has access)
-- =============================================================================
-- KGUPTA already has ACCOUNTADMIN role, so these are for completeness
GRANT USAGE ON WAREHOUSE DEV_WH TO ROLE ACCOUNTADMIN;
GRANT USAGE ON WAREHOUSE STAGING_WH TO ROLE ACCOUNTADMIN;
GRANT USAGE ON WAREHOUSE PROD_WH TO ROLE ACCOUNTADMIN;
GRANT USAGE ON WAREHOUSE ANALYTICS_WH TO ROLE ACCOUNTADMIN;

-- Grant Database Permissions to KGUPTA
-- =============================================================================
GRANT USAGE ON DATABASE METRIFY_OPERATIONAL TO ROLE ACCOUNTADMIN;
GRANT USAGE ON DATABASE METRIFY_ANALYTICS TO ROLE ACCOUNTADMIN;
GRANT USAGE ON DATABASE METRIFY_ARCHIVE TO ROLE ACCOUNTADMIN;

-- Grant Schema Permissions to KGUPTA
-- =============================================================================
-- METRIFY_OPERATIONAL schemas
GRANT USAGE ON ALL SCHEMAS IN DATABASE METRIFY_OPERATIONAL TO ROLE ACCOUNTADMIN;

-- METRIFY_ANALYTICS schemas
GRANT USAGE ON ALL SCHEMAS IN DATABASE METRIFY_ANALYTICS TO ROLE ACCOUNTADMIN;

-- METRIFY_ARCHIVE schemas
GRANT USAGE ON ALL SCHEMAS IN DATABASE METRIFY_ARCHIVE TO ROLE ACCOUNTADMIN;

-- Grant Table Permissions to KGUPTA
-- =============================================================================
-- METRIFY_OPERATIONAL tables
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN DATABASE METRIFY_OPERATIONAL TO ROLE ACCOUNTADMIN;

-- METRIFY_ANALYTICS tables
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN DATABASE METRIFY_ANALYTICS TO ROLE ACCOUNTADMIN;

-- METRIFY_ARCHIVE tables
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN DATABASE METRIFY_ARCHIVE TO ROLE ACCOUNTADMIN;

-- Grant Future Permissions to KGUPTA
-- =============================================================================
-- Future table permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON FUTURE TABLES IN DATABASE METRIFY_OPERATIONAL TO ROLE ACCOUNTADMIN;
GRANT SELECT, INSERT, UPDATE, DELETE ON FUTURE TABLES IN DATABASE METRIFY_ANALYTICS TO ROLE ACCOUNTADMIN;
GRANT SELECT, INSERT, UPDATE, DELETE ON FUTURE TABLES IN DATABASE METRIFY_ARCHIVE TO ROLE ACCOUNTADMIN;

-- Test the Setup
-- =============================================================================
-- Switch to the analytics warehouse and database
USE WAREHOUSE ANALYTICS_WH;
USE DATABASE METRIFY_ANALYTICS;
USE SCHEMA ANALYTICS;

-- Test query
SELECT 'Snowflake infrastructure setup completed successfully!' as status;
SELECT CURRENT_USER() as current_user;
SELECT CURRENT_ROLE() as current_role;
SELECT CURRENT_WAREHOUSE() as current_warehouse;
SELECT CURRENT_DATABASE() as current_database;
SELECT CURRENT_SCHEMA() as current_schema;
