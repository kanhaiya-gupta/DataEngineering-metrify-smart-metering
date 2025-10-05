# Metrify Smart Metering Data Pipeline v2

> **⚡ Powering Germany's Energy Transition with Real-Time Smart Meter Intelligence**

Transform millions of smart meters into actionable insights with our enterprise-grade data pipeline. Built for scale, security, and the future of decentralized energy grids.

> **⚠️ PROPRIETARY SOFTWARE**: This is proprietary software owned by Metrify. Unauthorized use, distribution, or modification is strictly prohibited. For licensing inquiries, contact licensing@metrify.com

## 🎯 Core System Overview

```mermaid
graph TB
    %% Data Sources
    subgraph "📊 Data Sources"
        SM[Smart Meters<br/>⚡ Real-time Readings]
        GO[Grid Operators<br/>🔌 TenneT, 50Hertz, Amprion]
        WS[Weather Stations<br/>🌤️ DWD, OpenWeatherMap]
    end

    %% Data Ingestion Layer
    subgraph "🌊 Data Ingestion"
        KAFKA[Apache Kafka<br/>📡 Real-time Streaming]
        AIRFLOW[Apache Airflow<br/>⏰ Workflow Orchestration]
    end

    %% Processing Layer
    subgraph "⚙️ Data Processing"
        API[FastAPI Services<br/>🚀 REST API]
        WORKERS[Background Workers<br/>🔄 Processing & Analytics]
        SPARK[Apache Spark<br/>⚡ ETL Processing]
        DBT[dbt Transformations<br/>📊 Data Modeling]
    end

    %% Storage Layer
    subgraph "💾 Data Storage"
        POSTGRES[(PostgreSQL<br/>🗄️ Operational DB)]
        SNOWFLAKE[(Snowflake<br/>❄️ Data Warehouse)]
        S3[(AWS S3<br/>☁️ Data Lake)]
    end

    %% Monitoring & Observability
    subgraph "📈 Monitoring & Observability"
        DATADOG[DataDog<br/>📊 Metrics & Logs]
        PROMETHEUS[Prometheus<br/>📈 Metrics Collection]
        GRAFANA[Grafana<br/>📊 Dashboards]
        JAEGER[Jaeger<br/>🔍 Distributed Tracing]
    end

    %% Infrastructure
    subgraph "🏗️ Infrastructure"
        DOCKER[Docker<br/>🐳 Containerization]
        K8S[Kubernetes<br/>☸️ Orchestration]
        TERRAFORM[Terraform<br/>🏗️ Infrastructure as Code]
    end

    %% Data Flow
    SM --> KAFKA
    GO --> KAFKA
    WS --> KAFKA
    
    KAFKA --> API
    KAFKA --> WORKERS
    KAFKA --> SPARK
    AIRFLOW --> WORKERS
    AIRFLOW --> SPARK
    
    API --> POSTGRES
    WORKERS --> POSTGRES
    WORKERS --> SNOWFLAKE
    WORKERS --> S3
    SPARK --> S3
    SPARK --> SNOWFLAKE
    
    DBT --> SNOWFLAKE
    
    %% Monitoring connections
    API --> DATADOG
    WORKERS --> DATADOG
    SPARK --> DATADOG
    KAFKA --> PROMETHEUS
    POSTGRES --> GRAFANA
    SNOWFLAKE --> GRAFANA
    
    %% Infrastructure connections
    API --> DOCKER
    WORKERS --> DOCKER
    SPARK --> DOCKER
    DOCKER --> K8S
    K8S --> TERRAFORM

    %% Styling
    classDef dataSource fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef processing fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef storage fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef monitoring fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef infrastructure fill:#fce4ec,stroke:#880e4f,stroke-width:2px

    class SM,GO,WS dataSource
    class KAFKA,AIRFLOW,API,WORKERS,SPARK,DBT processing
    class POSTGRES,SNOWFLAKE,S3 storage
    class DATADOG,PROMETHEUS,GRAFANA,JAEGER monitoring
    class DOCKER,K8S,TERRAFORM infrastructure
```

## 🚀 Comprehensive System Overview

*For a detailed view with all advanced features including ML/AI, Advanced Analytics, Data Governance, Performance Optimization, and Multi-Cloud Architecture, see the [Comprehensive System Overview](docs/README.md#comprehensive-system-overview).*

## 🚀 Key Features

### Core Capabilities
- **⚡ Real-time Processing**: Sub-second latency for critical smart meter data
- **📈 Scalable Architecture**: Handles millions of meters with horizontal scaling
- **🛡️ Enterprise Security**: End-to-end encryption, RBAC, and audit logging
- **🔍 Data Quality**: Automated validation, anomaly detection, and quality scoring
- **📊 Advanced Analytics**: ML-powered insights and predictive analytics
- **🌐 Multi-tenant**: Support for multiple energy providers and grid operators

### Energy Sector Specific
- **🔌 Smart Meter Integration**: Real-time consumption, voltage, and power factor monitoring
- **⚡ Grid Stability Analysis**: Multi-operator grid status monitoring and analysis
- **🌤️ Weather Correlation**: Energy demand forecasting based on weather patterns
- **🚨 Anomaly Detection**: Automated detection of meter malfunctions and data quality issues
- **📋 Compliance**: GDPR, energy sector regulations, and audit requirements

### Advanced Features
- **🤖 Machine Learning & AI**: TensorFlow-based forecasting, anomaly detection, and predictive maintenance
- **🏛️ Data Governance**: Apache Atlas catalog, data lineage, PII protection, and compliance automation
- **📈 Advanced Analytics**: Time series forecasting, multivariate analysis, and interactive visualizations
- **🔍 Data Quality Management**: Great Expectations validation, ML-based quality prediction, and automated remediation
- **⚡ Performance Optimization**: Multi-level caching, query optimization, and stream processing with Apache Flink
- **🔄 Hybrid ETL Architecture**: Apache Spark for heavy data processing + dbt for analytics and business logic
- **☁️ Multi-Cloud Architecture**: AWS, Azure, and GCP deployment with cross-cloud replication and failover

## ✨ Why Choose This Solution?

### 🎯 **Built for Scale**
- Handle **millions of smart meters** with sub-second latency
- **Horizontal scaling** across multiple data centers
- **Auto-scaling** based on real-time demand

### 🛡️ **Enterprise-Grade Security**
- **End-to-end encryption** for all data in transit and at rest
- **Role-based access control** with fine-grained permissions
- **Complete audit trail** for compliance and security

### 🔧 **Developer-Friendly**
- **Clean Architecture** for maintainable, testable code
- **Comprehensive documentation** and examples
- **Easy deployment** with Docker and Kubernetes

### ⚡ **Real-Time Intelligence**
- **Live data processing** from smart meters
- **ML-powered anomaly detection** for proactive maintenance
- **Weather correlation** for demand forecasting

## 🏗️ Clean Architecture

This solution implements a **Clean Architecture** pattern with clear separation of concerns, making it highly maintainable, testable, and scalable for enterprise environments.

### Architecture Principles

- **🏛️ Clean Architecture**: Clear separation between business logic, data access, and presentation layers
- **🎯 Domain-Driven Design**: Business logic centered around smart metering domain
- **🔧 SOLID Principles**: Single responsibility, open/closed, dependency inversion
- **📦 Repository Pattern**: Abstract data access layer for testability
- **🔄 CQRS**: Command Query Responsibility Segregation for complex operations
- **📊 Event Sourcing**: Complete audit trail and state management
- **💉 Dependency Injection**: Loose coupling and high testability

## 🏛️ Clean Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Presentation Layer                                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐   │
│  │   REST API  │ │     CLI     │ │ Background  │ │    ML/Analytics         │   │
│  │             │ │             │ │  Workers    │ │    Endpoints             │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Application Layer                                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐   │
│  │  Use Cases  │ │     DTOs    │ │   Event/    │ │    ML/Analytics         │   │
│  │             │ │             │ │  Command    │ │    Services              │   │
│  │             │ │             │ │  Handlers   │ │                          │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Core Layer                                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐   │
│  │   Domain    │ │  Services   │ │   ML/AI     │ │    Analytics            │   │
│  │   Models    │ │             │ │   Models    │ │    Engine               │   │
│  │             │ │             │ │             │ │                         │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────────────┘   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐   │
│  │  Interfaces │ │  Value      │ │  Quality    │ │    Performance          │   │
│  │  &          │ │  Objects    │ │  Models     │ │    Models               │   │
│  │  Exceptions │ │             │ │             │ │                         │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Infrastructure Layer                                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐   │
│  │  Database   │ │  External   │ │  Monitoring │ │    Multi-Cloud          │   │
│  │  & Repos    │ │  Services   │ │  & Logging  │ │    Infrastructure        │   │
│  │             │ │             │ │             │ │                         │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────────────┘   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐   │
│  │  Caching    │ │  Stream     │ │  Data       │ │    Performance          │   │
│  │  Layer      │ │  Processing │ │  Governance │ │    Optimization          │   │
│  │             │ │             │ │             │ │                         │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

This project follows **Clean Architecture** principles with clear separation of concerns:

- **`src/`** - Core business logic and infrastructure
  - **`core/`** - Domain models, services, and interfaces
  - **`infrastructure/`** - Database, external services, and monitoring
  - **`application/`** - Use cases, DTOs, and handlers
- **`presentation/`** - REST API, CLI tools, and background workers
- **`config/`** - Environment-specific configurations
- **`tests/`** - Comprehensive test suite (unit, integration, e2e, performance)
- **`infrastructure/`** - Docker, Kubernetes, and Terraform configurations
- **`docs/`** - Complete documentation and user guides
- **`dbt/`** - Data transformation models and tests

📋 **For detailed project structure, see [Project Structure](docs/project-structure.md)**

## 🛠️ Technology Stack

### Core Technologies
- **🐍 Python 3.11+**: Modern Python with type hints and async support
- **🏗️ FastAPI**: High-performance async web framework
- **🗄️ PostgreSQL 15+**: Primary operational database
- **❄️ Snowflake**: Data warehouse for analytics
- **☁️ AWS**: Cloud infrastructure and services

### Data Processing
- **🌊 Apache Kafka**: Real-time data streaming
- **⚡ Apache Flink**: Advanced stream processing
- **🔄 Apache Airflow**: Workflow orchestration
- **⚡ Apache Spark**: Distributed ETL processing
- **📊 dbt**: Data transformation and modeling

### Machine Learning & AI
- **🤖 TensorFlow**: Deep learning and ML models
- **📊 MLflow**: Model lifecycle management
- **🍽️ Feast**: Feature store and serving
- **🔍 Evidently**: Data drift detection
- **📈 SHAP**: Model explainability

### Data Governance & Quality
- **🏛️ Apache Atlas**: Data catalog and lineage
- **✅ Great Expectations**: Data validation framework
- **🔒 Apache Ranger**: Data security and access control
- **📊 Custom Quality Engine**: Business-specific validations

### Performance & Caching
- **⚡ Redis**: Multi-level caching
- **🚀 Apache Flink**: Stream processing optimization
- **📊 Custom Optimizer**: Query and index optimization
- **💾 Intelligent Caching**: Predictive cache warming

### Multi-Cloud Infrastructure
- **☁️ AWS**: Primary cloud (eu-central-1)
- **☁️ Azure**: Secondary cloud (Germany West Central)
- **☁️ GCP**: Tertiary cloud (europe-west3)
- **🔄 Cross-Cloud Replication**: Data synchronization and failover

### Infrastructure
- **🐳 Docker**: Containerization
- **☸️ Kubernetes**: Container orchestration
- **🏗️ Terraform**: Multi-cloud Infrastructure as Code
- **📈 DataDog**: Monitoring and observability

### Development & Testing
- **🧪 pytest**: Testing framework with async, coverage, and performance testing
- **📊 pytest-cov**: Code coverage analysis
- **🔍 mypy**: Static type checking
- **🖤 black**: Code formatting
- **📋 pre-commit**: Git hooks for code quality
- **📦 Single requirements.txt**: All dependencies in one comprehensive file

## 🚀 Get Started in Minutes

Ready to transform your smart meter data into actionable insights? Follow our quick start guide to get up and running in minutes.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- PostgreSQL 15+
- AWS CLI configured
- Snowflake account

### 1. Clone and Setup
```bash
git clone https://github.com/metrify/smart-metering-pipeline.git
cd smart-metering-pipeline
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt  # Installs all dependencies including dbt, testing, and development tools
```

### 2. Configuration
```bash
cp config/environments/development.yaml config/active.yaml
# Edit config/active.yaml with your settings
```

### 3. Database Setup
```bash
# Start PostgreSQL
docker-compose -f infrastructure/docker/docker-compose.yml up -d postgres

# Run migrations
python scripts/migrate.py --environment development
```

### 4. Start Services
```bash
# Start all services
docker-compose -f infrastructure/docker/docker-compose.yml up -d

# Or start specific services
python -m presentation.api.main  # Start API server
python -m presentation.workers.ingestion_worker  # Start ingestion worker
```

### 5. Verify Installation
```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs
```

## 🎯 What's Next?

### 🏃‍♂️ **Quick Wins**
- **Explore the API**: Visit `http://localhost:8000/docs` for interactive API documentation
- **Check Data Quality**: Run quality checks on your smart meter data
- **View Analytics**: Access real-time dashboards and reports

### 🚀 **Advanced Usage**
- **Custom Integrations**: Connect your existing systems via our REST API
- **Data Pipeline**: Set up automated data processing workflows
- **Monitoring**: Configure alerts and monitoring for your deployment

### 📚 **Learn More**
- **Architecture Deep Dive**: Understand the system design and components
- **API Reference**: Complete API documentation with examples
- **Deployment Guide**: Production deployment strategies and best practices

## 📚 Documentation

### 📁 Project Structure
- **[Project Structure](docs/project-structure.md)** - Complete project structure and file organization

### 🏗️ Architecture Documentation
- **[System Overview](docs/architecture/system-overview.md)** - High-level system architecture and components
- **[Data Flow](docs/architecture/data-flow.md)** - How data moves through the system
- **[Technology Stack](docs/architecture/technology-stack.md)** - Technologies and tools used
- **[Security Architecture](docs/architecture/security-architecture.md)** - Security design and implementation
- **[Scalability Design](docs/architecture/scalability-design.md)** - How the system scales

### 🔌 API Documentation
- **[API Overview](docs/api/api-overview.md)** - REST API introduction and authentication
- **[Smart Meter API](docs/api/smart-meter-api.md)** - Smart meter endpoints and operations
- **[Grid Operator API](docs/api/grid-operator-api.md)** - Grid operator endpoints and operations
- **[Weather API](docs/api/weather-api.md)** - Weather station endpoints and operations
- **[Analytics API](docs/api/analytics-api.md)** - Analytics and reporting endpoints
- **[API Reference](docs/api/api-reference.md)** - Complete API reference with examples

### 🚀 Deployment Documentation
- **[Deployment Overview](docs/deployment/deployment-overview.md)** - Deployment strategies and environments
- **[Local Development](docs/deployment/local-development.md)** - Setting up local development environment
- **[Docker Deployment](docs/deployment/docker-deployment.md)** - Containerized deployment
- **[Kubernetes Deployment](docs/deployment/kubernetes-deployment.md)** - Kubernetes orchestration
- **[Production Deployment](docs/deployment/production-deployment.md)** - Production deployment guide
- **[Monitoring Setup](docs/deployment/monitoring-setup.md)** - Monitoring and observability setup

### 👥 User Guides
- **[Getting Started](docs/user_guides/getting-started.md)** - Quick start guide for new users
- **[Data Ingestion Guide](docs/user_guides/data-ingestion-guide.md)** - How to ingest data into the system
- **[Data Quality Guide](docs/user_guides/data-quality-guide.md)** - Understanding and managing data quality
- **[Analytics Guide](docs/user_guides/analytics-guide.md)** - Using analytics and reporting features
- **[Troubleshooting Guide](docs/user_guides/troubleshooting-guide.md)** - Common issues and solutions
- **[Best Practices](docs/user_guides/best-practices.md)** - Recommended practices and guidelines

### 📊 Data Transformation (dbt)
- **[dbt Project Overview](dbt/README.md)** - dbt project structure and configuration
- **[Data Models](dbt/models/)** - Staging, marts, and metrics models
- **[Tests and Quality](dbt/tests/)** - Data quality tests and validations
- **[Macros](dbt/macros/)** - Reusable SQL functions and transformations

### ⚡ Spark ETL Integration
- **[Spark ETL Overview](roadmap/initiatives/spark-etl-integration/README.md)** - Spark ETL integration initiative and business justification
- **[Technical Implementation](roadmap/initiatives/spark-etl-integration/technical-implementation.md)** - Technical architecture and implementation details
- **[Migration Strategy](roadmap/initiatives/spark-etl-integration/migration-strategy.md)** - Migration approach and rollout plan

### 🤖 ML & Analytics Snowflake Integration
- **[ML-Snowflake Overview](roadmap/initiatives/ml-snowflake-integration/README.md)** - ML and Analytics integration with Snowflake initiative
- **[Technical Implementation](roadmap/initiatives/ml-snowflake-integration/technical-implementation.md)** - Technical architecture for ML/Analytics Snowflake integration
- **[Migration Strategy](roadmap/initiatives/ml-snowflake-integration/migration-strategy.md)** - Migration strategy from PostgreSQL to Snowflake for ML/Analytics

### 🤖 Machine Learning & AI
- **[ML Overview](docs/ml/ml-overview.md)** - Machine learning capabilities and architecture
- **[Model Training](docs/ml/model-training.md)** - ML model development and training
- **[Model Deployment](docs/ml/model-deployment.md)** - ML model serving and deployment
- **[ML Monitoring](docs/ml/ml-monitoring.md)** - ML model monitoring and management

### 🏛️ Data Governance
- **[Governance Overview](docs/governance/governance-overview.md)** - Data governance framework
- **[Data Catalog](docs/governance/data-catalog.md)** - Data discovery and cataloging
- **[Data Lineage](docs/governance/data-lineage.md)** - Data lineage tracking and visualization
- **[Privacy & Security](docs/governance/privacy-security.md)** - Data protection and compliance

### 📈 Advanced Analytics
- **[Analytics Overview](docs/analytics/analytics-overview.md)** - Advanced analytics capabilities
- **[Forecasting Guide](docs/analytics/forecasting-guide.md)** - Time series forecasting
- **[Anomaly Detection](docs/analytics/anomaly-detection-guide.md)** - Anomaly detection and analysis
- **[Visualization Guide](docs/analytics/visualization-guide.md)** - Interactive visualizations

### 🔍 Data Quality
- **[Quality Overview](docs/quality/quality-overview.md)** - Data quality management
- **[Quality Rules](docs/quality/quality-rules.md)** - Data quality validation rules
- **[Quality Monitoring](docs/quality/quality-monitoring.md)** - Quality monitoring and alerting
- **[Quality Automation](docs/quality/quality-automation.md)** - Automated quality processes

### ⚡ Performance Optimization
- **[Performance Overview](docs/performance/performance-overview.md)** - Performance optimization strategies
- **[Caching Guide](docs/performance/caching-guide.md)** - Multi-level caching implementation
- **[Query Optimization](docs/performance/query-optimization-guide.md)** - Database query optimization
- **[Stream Processing](docs/performance/stream-processing-guide.md)** - Real-time stream processing

### ☁️ Multi-Cloud Architecture
- **[Multi-Cloud Overview](docs/multicloud/multicloud-overview.md)** - Multi-cloud strategy and implementation
- **[AWS Deployment](docs/multicloud/aws-deployment.md)** - AWS cloud deployment guide
- **[Azure Deployment](docs/multicloud/azure-deployment.md)** - Azure cloud deployment guide
- **[GCP Deployment](docs/multicloud/gcp-deployment.md)** - Google Cloud deployment guide

## 🎯 Quick Navigation

### For Business Users
- Start with [Getting Started](docs/user_guides/getting-started.md)
- Learn about [Data Quality](docs/user_guides/data-quality-guide.md)
- Explore [Analytics Features](docs/user_guides/analytics-guide.md)

### For Developers
- Start with [Project Structure](docs/project-structure.md) to understand the codebase
- Review [System Architecture](docs/architecture/system-overview.md)
- Check [API Documentation](docs/api/api-overview.md)
- Follow [Local Development](docs/deployment/local-development.md)

### For DevOps Engineers
- Study [Deployment Overview](docs/deployment/deployment-overview.md)
- Configure [Monitoring](docs/deployment/monitoring-setup.md)
- Set up [Production Environment](docs/deployment/production-deployment.md)

### For Data Engineers
- Start with [Project Structure](docs/project-structure.md) to understand the data pipeline
- Understand [Data Flow](docs/architecture/data-flow.md)
- Learn [Data Ingestion](docs/user_guides/data-ingestion-guide.md)
- Review [Data Quality](docs/quality/quality-overview.md)
- Explore [dbt Transformations](dbt/README.md)
- Plan [Spark ETL Integration](roadmap/initiatives/spark-etl-integration/README.md)
- Integrate [ML & Analytics with Snowflake](roadmap/initiatives/ml-snowflake-integration/README.md)
- Master [Advanced Analytics](docs/analytics/analytics-overview.md)

### For ML Engineers
- Start with [ML Overview](docs/ml/ml-overview.md)
- Learn [Model Training](docs/ml/model-training.md)
- Deploy [Model Serving](docs/ml/model-deployment.md)
- Monitor [ML Performance](docs/ml/ml-monitoring.md)
- Integrate [ML with Snowflake](roadmap/initiatives/ml-snowflake-integration/README.md)

### For Data Governance Teams
- Understand [Governance Framework](docs/governance/governance-overview.md)
- Set up [Data Catalog](docs/governance/data-catalog.md)
- Track [Data Lineage](docs/governance/data-lineage.md)
- Ensure [Privacy & Security](docs/governance/privacy-security.md)

### For Performance Engineers
- Optimize [Performance](docs/performance/performance-overview.md)
- Implement [Caching](docs/performance/caching-guide.md)
- Tune [Query Performance](docs/performance/query-optimization-guide.md)
- Scale [Stream Processing](docs/performance/stream-processing-guide.md)

### For Cloud Architects
- Plan [Multi-Cloud Strategy](docs/multicloud/multicloud-overview.md)
- Deploy on [AWS](docs/multicloud/aws-deployment.md)
- Deploy on [Azure](docs/multicloud/azure-deployment.md)
- Deploy on [GCP](docs/multicloud/gcp-deployment.md)


## 📊 Visual Guides

This documentation includes comprehensive Mermaid diagrams to help visualize:

- **System Architecture** - How components interact
- **Data Flow** - How data moves through the system
- **User Workflows** - Step-by-step user processes
- **Deployment Processes** - How to deploy the system
- **Troubleshooting Flows** - How to resolve issues

## 🤝 Contributing

### Development Access
This is a **proprietary project** with restricted access. Only authorized Metrify team members and approved contributors may contribute to this repository.

### For Authorized Contributors
1. **Clone** the repository (access required)
2. **Create** a feature branch from `main`
3. **Write** tests for new functionality
4. **Implement** the feature following our standards
5. **Run** the test suite: `pytest`
6. **Submit** a pull request for review

### Code Standards
- **Type Hints**: All functions must have type hints
- **Documentation**: All public APIs must be documented
- **Testing**: Minimum 80% code coverage
- **Linting**: Code must pass all linting checks
- **Security**: Follow security best practices and guidelines

### External Contributions
If you're interested in contributing but don't have access, please contact:
- **Email**: contributions@metrify.com
- **Subject**: "Contribution Interest - Metrify Smart Metering Pipeline"

## 📄 License

This project is **proprietary software** owned by Metrify. All rights reserved.

### Usage Rights
- **Commercial Use**: Requires explicit written permission from Metrify
- **Distribution**: Prohibited without authorization
- **Modification**: Not permitted without prior consent
- **Private Use**: Limited to authorized personnel only

### Permission Requests
To use this software, please contact:
- **Email**: licensing@metrify.com
- **Subject**: "Metrify Smart Metering Data Pipeline - License Request"
- **Include**: Intended use case, organization details, and contact information

### Terms
By accessing this repository, you agree to:
1. Not use the software without proper authorization
2. Respect intellectual property rights
3. Contact Metrify for any licensing inquiries
4. Comply with all applicable laws and regulations

**© 2025 Metrify. All rights reserved.**

## 🆘 Support

- **Documentation**: [docs.metrify.com](https://docs.metrify.com)
- **Issues**: [GitHub Issues](https://github.com/metrify/smart-metering-pipeline/issues)
- **Discussions**: [GitHub Discussions](https://github.com/metrify/smart-metering-pipeline/discussions)
- **Email**: support@metrify.com

## 🙏 Acknowledgments

- **Enpal**: For the vision and support
- **Open Source Community**: For the amazing tools and libraries
- **Energy Sector**: For the domain expertise and requirements

---

**Built with ❤️ by the Metrify Data Engineering Team**

*Empowering the energy transition through data-driven insights*
