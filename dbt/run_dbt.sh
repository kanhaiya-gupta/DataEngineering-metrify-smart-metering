#!/bin/bash

# dbt run script for Metrify Smart Metering
# This script runs dbt commands with proper environment setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if dbt is installed
check_dbt() {
    if ! command -v dbt &> /dev/null; then
        print_error "dbt is not installed. Please install dbt first."
        exit 1
    fi
    print_success "dbt is installed: $(dbt --version)"
}

# Function to check if profiles.yml exists
check_profiles() {
    if [ ! -f ~/.dbt/profiles.yml ]; then
        print_warning "profiles.yml not found. Creating from project profiles.yml..."
        mkdir -p ~/.dbt
        cp profiles.yml ~/.dbt/profiles.yml
    fi
    print_success "dbt profiles configured"
}

# Function to install dbt packages
install_packages() {
    print_status "Installing dbt packages..."
    dbt deps
    print_success "dbt packages installed"
}

# Function to run dbt tests
run_tests() {
    local test_type=${1:-"all"}
    
    case $test_type in
        "all")
            print_status "Running all dbt tests..."
            dbt test
            ;;
        "staging")
            print_status "Running staging model tests..."
            dbt test --select staging
            ;;
        "marts")
            print_status "Running marts model tests..."
            dbt test --select marts
            ;;
        "metrics")
            print_status "Running metrics model tests..."
            dbt test --select metrics
            ;;
        *)
            print_error "Unknown test type: $test_type"
            exit 1
            ;;
    esac
    print_success "dbt tests completed"
}

# Function to run dbt models
run_models() {
    local model_type=${1:-"all"}
    local target=${2:-"dev"}
    
    case $model_type in
        "all")
            print_status "Running all dbt models..."
            dbt run --target $target
            ;;
        "staging")
            print_status "Running staging models..."
            dbt run --select staging --target $target
            ;;
        "marts")
            print_status "Running marts models..."
            dbt run --select marts --target $target
            ;;
        "metrics")
            print_status "Running metrics models..."
            dbt run --select metrics --target $target
            ;;
        *)
            print_error "Unknown model type: $model_type"
            exit 1
            ;;
    esac
    print_success "dbt models completed"
}

# Function to run dbt seeds
run_seeds() {
    print_status "Running dbt seeds..."
    dbt seed
    print_success "dbt seeds completed"
}

# Function to generate dbt docs
generate_docs() {
    print_status "Generating dbt documentation..."
    dbt docs generate
    print_success "dbt documentation generated"
}

# Function to run dbt compile
compile_models() {
    print_status "Compiling dbt models..."
    dbt compile
    print_success "dbt models compiled"
}

# Function to run dbt parse
parse_models() {
    print_status "Parsing dbt models..."
    dbt parse
    print_success "dbt models parsed"
}

# Function to run dbt clean
clean_project() {
    print_status "Cleaning dbt project..."
    dbt clean
    print_success "dbt project cleaned"
}

# Function to run dbt debug
debug_project() {
    print_status "Debugging dbt project..."
    dbt debug
    print_success "dbt project debug completed"
}

# Function to show dbt help
show_help() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  install     Install dbt packages"
    echo "  parse       Parse dbt models"
    echo "  compile     Compile dbt models"
    echo "  run         Run dbt models [all|staging|marts|metrics] [target]"
    echo "  test        Run dbt tests [all|staging|marts|metrics]"
    echo "  seed        Run dbt seeds"
    echo "  docs        Generate dbt documentation"
    echo "  clean       Clean dbt project"
    echo "  debug       Debug dbt project"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 install"
    echo "  $0 run staging dev"
    echo "  $0 test marts"
    echo "  $0 run all prod"
    echo "  $0 docs"
}

# Main script logic
main() {
    # Check if dbt is installed
    check_dbt
    
    # Check profiles
    check_profiles
    
    # Parse command line arguments
    case ${1:-"help"} in
        "install")
            install_packages
            ;;
        "parse")
            parse_models
            ;;
        "compile")
            compile_models
            ;;
        "run")
            run_models ${2:-"all"} ${3:-"dev"}
            ;;
        "test")
            run_tests ${2:-"all"}
            ;;
        "seed")
            run_seeds
            ;;
        "docs")
            generate_docs
            ;;
        "clean")
            clean_project
            ;;
        "debug")
            debug_project
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
