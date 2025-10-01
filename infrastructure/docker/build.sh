#!/bin/bash
# Docker build script for Metrify Smart Metering Data Pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REGISTRY=${DOCKER_REGISTRY:-"metrify"}
VERSION=${VERSION:-"latest"}
ENVIRONMENT=${ENVIRONMENT:-"development"}

echo -e "${BLUE}🐳 Building Metrify Smart Metering Docker Images${NC}"
echo "=================================================="
echo "Registry: $REGISTRY"
echo "Version: $VERSION"
echo "Environment: $ENVIRONMENT"
echo "=================================================="

# Function to build and tag image
build_image() {
    local service=$1
    local dockerfile=$2
    local context=$3
    
    echo -e "${YELLOW}Building $service...${NC}"
    
    # Build the image
    docker build \
        -f "$dockerfile" \
        -t "$REGISTRY/$service:$VERSION" \
        -t "$REGISTRY/$service:latest" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VERSION="$VERSION" \
        --build-arg ENVIRONMENT="$ENVIRONMENT" \
        "$context"
    
    echo -e "${GREEN}✅ $service built successfully${NC}"
}

# Function to push image
push_image() {
    local service=$1
    
    if [ "$PUSH_IMAGES" = "true" ]; then
        echo -e "${YELLOW}Pushing $service...${NC}"
        docker push "$REGISTRY/$service:$VERSION"
        docker push "$REGISTRY/$service:latest"
        echo -e "${GREEN}✅ $service pushed successfully${NC}"
    fi
}

# Build API service
build_image "metrify-api" "Dockerfile.api" "../.."
push_image "metrify-api"

# Build Worker service
build_image "metrify-worker" "Dockerfile.worker" "../.."
push_image "metrify-worker"

echo -e "${GREEN}🎉 All images built successfully!${NC}"

# Show built images
echo -e "${BLUE}📋 Built Images:${NC}"
docker images | grep "$REGISTRY" | head -10

# Show usage instructions
echo -e "${BLUE}📖 Usage Instructions:${NC}"
echo "1. Development: docker-compose up -d"
echo "2. Production: docker-compose -f docker-compose.prod.yml up -d"
echo "3. Scale workers: docker-compose up -d --scale ingestion-worker=3"
echo "4. View logs: docker-compose logs -f [service-name]"
echo "5. Stop services: docker-compose down"

# Health check
if [ "$HEALTH_CHECK" = "true" ]; then
    echo -e "${YELLOW}🔍 Running health checks...${NC}"
    
    # Start services for health check
    docker-compose up -d postgres redis kafka
    
    # Wait for services to be ready
    echo "Waiting for services to be ready..."
    sleep 30
    
    # Check API health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ API health check passed${NC}"
    else
        echo -e "${RED}❌ API health check failed${NC}"
    fi
    
    # Cleanup
    docker-compose down
fi

echo -e "${GREEN}🚀 Build completed successfully!${NC}"
