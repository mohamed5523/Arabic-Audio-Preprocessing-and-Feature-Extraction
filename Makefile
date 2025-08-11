# Makefile for Arabic-Audio-Preprocessing-and-Feature-Extraction

.PHONY: help install install-dev setup clean test lint format run docker-build docker-run docker-stop milvus-up milvus-down

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install dependencies"
	@echo "  install-dev  - Install dev dependencies"
	@echo "  setup        - Setup virtual environment and install dependencies"
	@echo "  clean        - Clean build artifacts"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  run          - Run the Arabic-Audio-Preprocessing-and-Feature-Extraction"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run with Docker"
	@echo "  docker-stop  - Stop Docker containers"
	@echo "  milvus-up    - Start Milvus with docker-compose"
	@echo "  milvus-down  - Stop Milvus"

# Install dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -r requirements.txt
	pip install pytest black flake8 isort

# Setup virtual environment
setup:
	python -m venv venv
	@echo "Activate virtual environment with:"
	@echo "  source venv/bin/activate  (Linux/Mac)"
	@echo "  venv\\Scripts\\activate     (Windows)"
	@echo "Then run: make install"

# Clean build artifacts
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage

# Run tests (if you add test files)
test:
	python -m pytest tests/ -v

# Run linting
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Format code
format:
	black .
	isort .

# Run the Arabic-Audio-Preprocessing-and-Feature-Extractione
run:
	python main.py

# Docker targets
docker-build:
	docker build -t Arabic-Audio-Preprocessing-and-Feature-Extraction .

docker-run:
	docker run -it --rm \
		-v $(PWD)/input:/app/input \
		-v $(PWD)/output:/app/output \
		-e HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} \
		--network host \
		Arabic-Audio-Preprocessing-and-Feature-Extraction

docker-stop:
	docker stop $$(docker ps -q --filter ancestor=Arabic-Audio-Preprocessing-and-Feature-Extraction) || true

# Milvus targets
milvus-up:
	docker-compose up -d
	@echo "Waiting for Milvus to be ready..."
	@sleep 30
	@echo "Milvus is ready at http://localhost:19530"
	@echo "Attu (Web UI) is available at http://localhost:18080"

milvus-down:
	docker-compose down

# Check Milvus status
milvus-status:
	docker-compose ps

# View Milvus logs
milvus-logs:
	docker-compose logs -f milvus-standalone

# Install package in development mode
install-editable:
	pip install -e .

# Create requirements.txt from current environment
freeze:
	pip freeze > requirements.txt

# Check code style
check:
	black --check .
	isort --check-only .
	flake8 .