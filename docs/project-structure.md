# Project Structure

This document provides a comprehensive overview of the Metrify Smart Metering project structure, including all directories, files, and their purposes. This is your roadmap to understanding the entire codebase organization.

## 🎯 Project Overview

The Metrify Smart Metering project follows a clean architecture pattern with clear separation of concerns, making it maintainable, scalable, and easy to understand.

## 📁 Root Directory Structure

```
DataEngineering/
├── src/                          # Core business logic (Clean Architecture)
├── presentation/                 # Presentation layer (API, CLI, Workers)
├── tests/                       # Comprehensive test suite
├── config/                      # Configuration management
├── infrastructure/              # Infrastructure as Code (Docker, K8s, Terraform)
├── dbt/                         # Data transformations and modeling
├── docs/                        # Comprehensive documentation
├── requirements.txt             # Python dependencies
├── production.env              # Production environment variables
├── development.env             # Development environment variables
├── env.example                 # Environment variables template
└── README.md                   # Project overview and getting started
```

## 🏗️ Architecture Visualization

```mermaid
graph TB
    subgraph "Presentation Layer"
        A[API Endpoints] --> B[FastAPI REST API]
        C[CLI Tools] --> D[Command Line Interfaces]
        E[Background Workers] --> F[Processing Services]
    end
    
    subgraph "Core Business Logic"
        G[Domain Layer] --> H[Entities & Value Objects]
        I[Application Layer] --> J[Use Cases & DTOs]
        K[Infrastructure Layer] --> L[External Services]
    end
    
    subgraph "Data Layer"
        M[PostgreSQL] --> N[Operational Database]
        O[AWS S3] --> P[Data Lake]
        Q[Snowflake] --> R[Data Warehouse]
    end
    
    subgraph "Processing Layer"
        S[Apache Kafka] --> T[Message Streaming]
        U[Apache Airflow] --> V[Workflow Orchestration]
        W[dbt] --> X[Data Transformation]
    end
    
    subgraph "Infrastructure"
        Y[Docker] --> Z[Containerization]
        AA[Kubernetes] --> BB[Container Orchestration]
        CC[Terraform] --> DD[Infrastructure as Code]
    end
    
    A --> G
    C --> G
    E --> G
    G --> M
    I --> O
    K --> Q
    S --> U
    U --> W
    Y --> AA
    AA --> CC
```

## 📂 Detailed Directory Structure

### 1. Core Business Logic (`src/`)

```mermaid
graph TB
    subgraph "src/ - Core Business Logic"
        A[core/] --> B[Domain Layer]
        C[application/] --> D[Application Layer]
        E[infrastructure/] --> F[Infrastructure Layer]
    end
    
    B --> G[entities/]
    B --> H[value_objects/]
    B --> I[enums/]
    B --> J[events/]
    B --> K[services/]
    B --> L[interfaces/]
    B --> M[exceptions/]
    B --> N[config/]
    
    D --> O[use_cases/]
    D --> P[dto/]
    D --> Q[handlers/]
    
    E --> R[database/]
    E --> S[external/]
```

#### Domain Layer (`src/core/`)
```
src/core/
├── domain/
│   ├── entities/                 # Business entities
│   │   ├── smart_meter.py       # Smart meter aggregate root
│   │   ├── grid_operator.py     # Grid operator entity
│   │   └── weather_station.py   # Weather station entity
│   ├── value_objects/           # Immutable value objects
│   │   ├── meter_id.py          # Meter identifier
│   │   ├── location.py          # Geographic location
│   │   ├── meter_specifications.py
│   │   └── quality_score.py     # Data quality score
│   ├── enums/                   # Business enumerations
│   │   ├── meter_status.py      # Meter status types
│   │   ├── quality_tier.py      # Quality classification
│   │   ├── alert_level.py       # Alert severity levels
│   │   └── weather_station_status.py
│   ├── events/                  # Domain events
│   │   ├── meter_events.py      # Smart meter events
│   │   ├── grid_events.py       # Grid operator events
│   │   └── weather_events.py    # Weather station events
│   ├── services/                # Domain services
│   │   ├── smart_meter_service.py
│   │   ├── grid_operator_service.py
│   │   └── weather_service.py
│   ├── interfaces/              # Repository interfaces
│   │   ├── repositories/        # Data access contracts
│   │   └── external/            # External service contracts
│   ├── exceptions/              # Domain exceptions
│   │   └── domain_exceptions.py
│   └── config/                  # Configuration management
│       └── config_loader.py     # Configuration loader
└── __init__.py
```

#### Application Layer (`src/application/`)
```
src/application/
├── use_cases/                   # Business use cases
│   ├── ingest_smart_meter_data.py
│   ├── process_grid_status.py
│   ├── analyze_weather_impact.py
│   └── detect_anomalies.py
├── dto/                        # Data Transfer Objects
│   ├── smart_meter_dto.py
│   ├── grid_status_dto.py
│   └── weather_dto.py
├── handlers/                   # Event and command handlers
│   ├── event_handlers/
│   └── command_handlers/
└── __init__.py
```

#### Infrastructure Layer (`src/infrastructure/`)
```
src/infrastructure/
├── database/                   # Database components
│   ├── models/                 # SQLAlchemy models
│   ├── repositories/           # Repository implementations
│   ├── config.py              # Database configuration
│   ├── migrations/            # Database migrations
│   └── schemas/               # Database schemas
├── external/                   # External service integrations
│   ├── kafka/                 # Kafka messaging
│   ├── s3/                    # AWS S3 storage
│   ├── apis/                  # External API services
│   ├── snowflake/             # Snowflake data warehouse
│   ├── monitoring/            # Monitoring services
│   └── airflow/               # Airflow orchestration
└── __init__.py
```

### 2. Presentation Layer (`presentation/`)

```mermaid
graph TB
    subgraph "presentation/ - User Interfaces"
        A[api/] --> B[REST API]
        C[cli/] --> D[Command Line Tools]
        E[workers/] --> F[Background Workers]
    end
    
    B --> G[v1/]
    B --> H[middleware/]
    B --> I[schemas/]
    
    D --> J[data_ingestion_cli.py]
    D --> K[quality_check_cli.py]
    D --> L[maintenance_cli.py]
    
    F --> M[ingestion_worker.py]
    F --> N[processing_worker.py]
    F --> O[monitoring_worker.py]
```

```
presentation/
├── api/                        # REST API endpoints
│   ├── v1/                    # API version 1
│   │   ├── smart_meter_endpoints.py
│   │   ├── grid_operator_endpoints.py
│   │   ├── weather_endpoints.py
│   │   └── analytics_endpoints.py
│   ├── middleware/            # API middleware
│   │   ├── auth_middleware.py
│   │   ├── logging_middleware.py
│   │   └── monitoring_middleware.py
│   ├── schemas/               # API request/response schemas
│   │   └── common.py
│   └── main.py               # FastAPI application
├── cli/                       # Command line interfaces
│   ├── data_ingestion_cli.py
│   ├── quality_check_cli.py
│   └── maintenance_cli.py
├── workers/                   # Background workers
│   ├── ingestion_worker.py
│   ├── processing_worker.py
│   └── monitoring_worker.py
└── __init__.py
```

### 3. Test Suite (`tests/`)

```mermaid
graph TB
    subgraph "tests/ - Comprehensive Testing"
        A[unit/] --> B[Unit Tests]
        C[integration/] --> D[Integration Tests]
        E[e2e/] --> F[End-to-End Tests]
        G[performance/] --> H[Performance Tests]
        I[utils/] --> J[Test Utilities]
    end
    
    B --> K[core/]
    B --> L[application/]
    B --> M[infrastructure/]
    B --> N[presentation/]
    
    D --> O[database/]
    D --> P[external_apis/]
    D --> Q[kafka/]
    
    F --> R[test_data_pipeline_e2e.py]
    F --> S[test_api_e2e.py]
    
    H --> T[test_load_tests.py]
    H --> U[test_stress_tests.py]
```

```
tests/
├── unit/                      # Unit tests
│   ├── core/                 # Domain layer tests
│   ├── application/          # Application layer tests
│   ├── infrastructure/       # Infrastructure layer tests
│   └── presentation/         # Presentation layer tests
├── integration/              # Integration tests
│   ├── database/            # Database integration tests
│   ├── external_apis/       # External API tests
│   └── kafka/               # Kafka integration tests
├── e2e/                     # End-to-end tests
│   ├── test_data_pipeline_e2e.py
│   └── test_api_e2e.py
├── performance/             # Performance tests
│   ├── test_load_tests.py
│   └── test_stress_tests.py
├── utils/                   # Test utilities
│   └── test_helpers.py
├── conftest.py             # Pytest configuration
├── pytest.ini             # Pytest settings
├── run_tests.py            # Test runner script
└── requirements-test.txt   # Test dependencies
```

### 4. Configuration Management (`config/`)

```mermaid
graph TB
    subgraph "config/ - Configuration Management"
        A[environments/] --> B[Environment Configs]
        C[database/] --> D[Database Configs]
        E[external_services/] --> F[Service Configs]
        G[monitoring/] --> H[Monitoring Configs]
    end
    
    B --> I[development.yaml]
    B --> J[staging.yaml]
    B --> K[production.yaml]
    
    D --> L[connection_pools.yaml]
    D --> M[query_optimization.yaml]
    
    E --> N[kafka.yaml]
    E --> O[s3.yaml]
    E --> P[snowflake.yaml]
    
    H --> Q[prometheus.yml]
    H --> R[grafana/]
    H --> S[jaeger.yml]
```

```
config/
├── environments/            # Environment-specific configurations
│   ├── development.yaml
│   ├── staging.yaml
│   └── production.yaml
├── database/               # Database configurations
│   ├── connection_pools.yaml
│   └── query_optimization.yaml
├── external_services/      # External service configurations
│   ├── kafka.yaml
│   ├── s3.yaml
│   └── snowflake.yaml
├── monitoring/             # Monitoring configurations
│   ├── prometheus.yml
│   ├── grafana/
│   └── jaeger.yml
├── config_loader.py        # Configuration loader utility
├── validate_config.py      # Configuration validator
└── generate_config.py      # Configuration generator
```

### 5. Infrastructure as Code (`infrastructure/`)

```mermaid
graph TB
    subgraph "infrastructure/ - Infrastructure as Code"
        A[docker/] --> B[Containerization]
        C[kubernetes/] --> D[Container Orchestration]
        E[terraform/] --> F[Infrastructure Provisioning]
    end
    
    B --> G[Dockerfile.api]
    B --> H[Dockerfile.worker]
    B --> I[docker-compose.yml]
    B --> J[docker-compose.prod.yml]
    
    C --> K[deployments/]
    C --> L[services/]
    C --> M[configmaps/]
    C --> N[namespaces/]
    
    E --> O[main.tf]
    E --> P[variables.tf]
    E --> Q[outputs.tf]
    E --> R[environments/]
```

```
infrastructure/
├── docker/                 # Docker containerization
│   ├── Dockerfile.api
│   ├── Dockerfile.worker
│   ├── docker-compose.yml
│   ├── docker-compose.prod.yml
│   ├── .dockerignore
│   └── build.sh
├── kubernetes/             # Kubernetes orchestration
│   ├── deployments/
│   ├── services/
│   ├── configmaps/
│   ├── namespaces/
│   └── deploy.sh
└── terraform/              # Infrastructure provisioning
    ├── main.tf
    ├── variables.tf
    ├── outputs.tf
    ├── environments/
    └── deploy.sh
```

### 6. Data Transformations (`dbt/`)

```mermaid
graph TB
    subgraph "dbt/ - Data Transformations"
        A[models/] --> B[Data Models]
        C[macros/] --> D[Reusable Logic]
        E[tests/] --> F[Data Quality Tests]
        G[seeds/] --> H[Reference Data]
    end
    
    B --> I[staging/]
    B --> J[marts/]
    B --> K[metrics/]
    
    D --> L[data_quality_checks.sql]
    D --> M[business_logic.sql]
    D --> N[data_transformations.sql]
    
    I --> O[stg_smart_meter_readings.sql]
    I --> P[stg_grid_status.sql]
    I --> Q[stg_weather_data.sql]
    
    J --> R[fct_smart_meter_analytics.sql]
    J --> S[dim_meters.sql]
    J --> T[dim_grid_operators.sql]
    J --> U[dim_weather_stations.sql]
```

```
dbt/
├── models/                 # Data transformation models
│   ├── staging/           # Staging layer models
│   │   ├── stg_smart_meter_readings.sql
│   │   ├── stg_grid_status.sql
│   │   ├── stg_weather_data.sql
│   │   └── schema.yml
│   ├── marts/             # Business logic models
│   │   ├── fct_smart_meter_analytics.sql
│   │   ├── dim_meters.sql
│   │   ├── dim_grid_operators.sql
│   │   ├── dim_weather_stations.sql
│   │   └── schema.yml
│   └── metrics/           # Aggregated metrics
│       ├── daily_consumption_metrics.sql
│       └── schema.yml
├── macros/                # Reusable SQL macros
│   ├── data_quality_checks.sql
│   ├── business_logic.sql
│   └── data_transformations.sql
├── seeds/                 # Reference data
│   └── seed_smart_meter_types.csv
├── dbt_project.yml        # dbt project configuration
├── profiles.yml           # Database connection profiles
├── packages.yml           # dbt package dependencies
├── requirements.txt       # Python dependencies
├── run_dbt.sh            # dbt execution script
└── README.md             # dbt documentation
```

### 7. Documentation (`docs/`)

```mermaid
graph TB
    subgraph "docs/ - Comprehensive Documentation"
        A[architecture/] --> B[System Architecture]
        C[api/] --> D[API Documentation]
        E[deployment/] --> F[Deployment Guides]
        G[user_guides/] --> H[User Guides]
    end
    
    B --> I[system-overview.md]
    B --> J[data-flow.md]
    B --> K[technology-stack.md]
    
    C --> L[api-overview.md]
    C --> M[smart-meter-api.md]
    C --> N[grid-operator-api.md]
    
    E --> O[deployment-overview.md]
    E --> P[local-development.md]
    E --> Q[docker-deployment.md]
    
    G --> R[getting-started.md]
    G --> S[data-quality-guide.md]
    G --> T[analytics-guide.md]
```

```
docs/
├── README.md              # Documentation index
├── project-structure.md   # This file
├── architecture/          # System architecture documentation
│   ├── system-overview.md
│   ├── data-flow.md
│   ├── technology-stack.md
│   ├── security-architecture.md
│   └── scalability-design.md
├── api/                   # API documentation
│   ├── api-overview.md
│   ├── smart-meter-api.md
│   ├── grid-operator-api.md
│   ├── weather-api.md
│   ├── analytics-api.md
│   └── api-reference.md
├── deployment/            # Deployment documentation
│   ├── deployment-overview.md
│   ├── local-development.md
│   ├── docker-deployment.md
│   ├── kubernetes-deployment.md
│   ├── production-deployment.md
│   └── monitoring-setup.md
└── user_guides/           # User guides
    ├── getting-started.md
    ├── data-ingestion-guide.md
    ├── data-quality-guide.md
    ├── analytics-guide.md
    ├── troubleshooting-guide.md
    └── best-practices.md
```

## 🔄 Data Flow Through Project Structure

```mermaid
flowchart TD
    A[Raw Data Sources] --> B[presentation/api/]
    B --> C[src/application/]
    C --> D[src/core/]
    D --> E[src/infrastructure/]
    E --> F[PostgreSQL/S3/Snowflake]
    F --> G[dbt/models/]
    G --> H[Analytics & Reports]
    
    I[presentation/workers/] --> J[Background Processing]
    J --> K[src/infrastructure/external/]
    K --> L[Kafka/Airflow/Monitoring]
    
    M[presentation/cli/] --> N[Management Operations]
    N --> O[config/]
    O --> P[Environment Configuration]
    
    Q[tests/] --> R[Quality Assurance]
    R --> S[All Components]
    
    T[infrastructure/] --> U[Deployment]
    U --> V[Docker/K8s/Terraform]
    
    W[docs/] --> X[Documentation]
    X --> Y[All Users]
```

## 🎯 Key Design Principles

### 1. Clean Architecture
- **Dependency Inversion**: Core business logic doesn't depend on external frameworks
- **Separation of Concerns**: Each layer has a specific responsibility
- **Testability**: Easy to unit test business logic in isolation

### 2. Domain-Driven Design (DDD)
- **Entities**: Smart meters, grid operators, weather stations
- **Value Objects**: Meter IDs, locations, quality scores
- **Aggregates**: Smart meter as aggregate root
- **Domain Events**: Business events for system integration

### 3. SOLID Principles
- **Single Responsibility**: Each class has one reason to change
- **Open/Closed**: Open for extension, closed for modification
- **Liskov Substitution**: Derived classes are substitutable for base classes
- **Interface Segregation**: Clients depend only on interfaces they use
- **Dependency Inversion**: Depend on abstractions, not concretions

### 4. Configuration Management
- **Environment-specific**: Different configs for dev/staging/prod
- **Type-safe**: Pydantic models for configuration validation
- **Centralized**: Single source of truth for all configurations
- **Secure**: Sensitive data in environment variables

## 🚀 Getting Started with the Project

### For Developers
1. **Start with**: `docs/architecture/system-overview.md`
2. **Understand**: `src/core/` domain layer
3. **Explore**: `presentation/api/` for API endpoints
4. **Test**: `tests/` for understanding behavior
5. **Deploy**: `infrastructure/` for deployment

### For Data Engineers
1. **Start with**: `dbt/` for data transformations
2. **Understand**: `src/infrastructure/database/` for data models
3. **Explore**: `config/` for data pipeline configuration
4. **Monitor**: `src/infrastructure/external/monitoring/`

### For DevOps Engineers
1. **Start with**: `infrastructure/` for deployment
2. **Understand**: `config/` for environment management
3. **Explore**: `docs/deployment/` for deployment guides
4. **Monitor**: `src/infrastructure/external/monitoring/`

### For Business Users
1. **Start with**: `docs/user_guides/getting-started.md`
2. **Understand**: `presentation/api/` for data access
3. **Explore**: `docs/user_guides/analytics-guide.md`
4. **Monitor**: `docs/user_guides/data-quality-guide.md`

## 📊 Project Statistics

| Component | Files | Lines of Code | Purpose |
|-----------|-------|---------------|---------|
| **Core Domain** | 25+ | 2,500+ | Business logic and entities |
| **Application** | 15+ | 1,800+ | Use cases and DTOs |
| **Infrastructure** | 40+ | 4,200+ | External integrations |
| **Presentation** | 20+ | 3,100+ | API and CLI interfaces |
| **Tests** | 30+ | 2,800+ | Comprehensive test coverage |
| **Configuration** | 15+ | 1,200+ | Environment management |
| **Infrastructure** | 25+ | 1,500+ | Deployment automation |
| **dbt** | 20+ | 2,000+ | Data transformations |
| **Documentation** | 15+ | 3,000+ | Comprehensive guides |
| **Total** | **200+** | **22,100+** | **Complete system** |

## 🔧 Development Workflow

```mermaid
graph LR
    A[Code Changes] --> B[Unit Tests]
    B --> C[Integration Tests]
    C --> D[Build & Package]
    D --> E[Deploy to Staging]
    E --> F[E2E Tests]
    F --> G[Deploy to Production]
    G --> H[Monitor & Alert]
    
    B --> I[src/core/]
    B --> J[src/application/]
    B --> K[src/infrastructure/]
    B --> L[presentation/]
    
    C --> M[database/]
    C --> N[external_apis/]
    C --> O[kafka/]
    
    F --> P[api_e2e/]
    F --> Q[data_pipeline_e2e/]
    
    H --> R[prometheus/]
    H --> S[grafana/]
    H --> T[jaeger/]
```

## 📞 Support and Resources

### Documentation
- **Project Structure**: This file (`docs/project-structure.md`)
- **Architecture**: `docs/architecture/`
- **API Reference**: `docs/api/`
- **User Guides**: `docs/user_guides/`

### Development Tools
- **IDE Configuration**: `.vscode/` (if present)
- **Linting**: `pyproject.toml` or similar
- **Testing**: `pytest` with comprehensive coverage
- **Documentation**: Mermaid diagrams throughout

### Quick Commands
```bash
# Run tests
python tests/run_tests.py all

# Start development environment
docker-compose up -d

# Run dbt transformations
cd dbt && ./run_dbt.sh run

# Generate documentation
cd docs && python -m mkdocs serve
```

This project structure provides a solid foundation for a scalable, maintainable, and well-documented smart metering data pipeline system. Each component has a clear purpose and follows established software engineering best practices.
