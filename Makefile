.PHONY: help up down build train feast test lint deploy logs

# ─────────────────────────────────────────────────────────────────────────────
help:
	@echo "Hospital Readmission Prediction — dev commands"
	@echo ""
	@echo "  make up          Start all services (local dev)"
	@echo "  make down        Stop and remove containers"
	@echo "  make build       Rebuild all images"
	@echo "  make train       Run training job (needs GPU + running infra)"
	@echo "  make feast       Run Feast materialization"
	@echo "  make test        Run all tests"
	@echo "  make lint        Run linters"
	@echo "  make deploy      Apply K3s manifests (needs kubeconfig)"
	@echo "  make logs        Tail API logs"

# ── Local dev ─────────────────────────────────────────────────────────────────
up:
	@cp -n .env.example .env 2>/dev/null || true
	docker compose up -d

down:
	docker compose down

build:
	docker compose build

# ── One-off jobs ──────────────────────────────────────────────────────────────
train:
	docker compose --profile training run --rm training

feast:
	docker compose --profile feast run --rm feast

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-adversarial:
	pytest tests/adversarial/ -v

test-load:
	k6 run tests/load/load_test.js

# ── Linting ───────────────────────────────────────────────────────────────────
lint:
	ruff check services/ training/ feature_store/ fairness/
	cd frontend && npm run lint

# ── Kubernetes / K3s ──────────────────────────────────────────────────────────
deploy:
	kubectl apply -f infra/k8s/namespace.yaml
	kubectl apply -f infra/k8s/configmap.yaml
	kubectl apply -f infra/k8s/postgres-statefulset.yaml
	kubectl apply -f infra/k8s/redis-deployment.yaml
	kubectl apply -f infra/k8s/mlflow-deployment.yaml
	kubectl apply -f infra/k8s/api-deployment.yaml
	kubectl apply -f infra/k8s/frontend-deployment.yaml
	kubectl apply -f infra/k8s/monitoring-deployment.yaml
	kubectl apply -f infra/k8s/ingress.yaml
	kubectl apply -f infra/k8s/training-cronjob.yaml

k8s-status:
	kubectl get all -n readmission

# ── Utilities ─────────────────────────────────────────────────────────────────
logs:
	docker compose logs -f api

generate-keys:
	@python -c "import secrets; print('SECRET_KEY=' + secrets.token_hex(32))"
	@python -c "from cryptography.fernet import Fernet; print('ENCRYPTION_KEY=' + Fernet.generate_key().decode())"
