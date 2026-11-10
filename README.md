# FleetReliability-AI 🚜⚙️

**Predictive Maintenance & Condition-Based Monitoring (CBM) System for Heavy Industrial & Mining Equipment**  
*Machine Learning Zoomcamp — Mid-Term Project*

---

## 1. Executive Summary & Business Problem

Unplanned downtime in critical mining and industrial machinery (e.g., SAG mill pumps, crushers, haulage systems) leads to severe operational bottlenecks and financial losses exceeding tens of thousands of dollars per hour. However, relying on naive binary classification models (`0` vs `1` at an arbitrary `0.5` threshold) is operationally unfeasible in production environments, as false alarms can trigger costly, unnecessary shutdowns.

**FleetReliability-AI** bridges the gap between statistical machine learning and real-world Asset Management (RCM / CBM):
- **Physics-Informed Feature Engineering:** Incorporates thermodynamic and mechanical domain laws (mechanical power $P_{\text{mech}}$, thermal dissipation gradients $\Delta T$, and cumulative overstrain indices) derived from real-time sensor telemetry.
- **Stochastic Cost Optimization (Monte Carlo Simulation):** Replaces static threshold rules with a Monte Carlo simulation of uncertain repair times ($\text{MTTR} \sim \text{LogNormal}$), asset criticality tiers ($L, M, H$), and non-destructive inspection costs ($C_{\text{FP}} \sim \text{Normal}$), minimizing both Expected Operational Loss and tail risk ($P_{90}$ Value-at-Risk).
- **Multi-Threshold Operational Action Policy:** Exposes a calibrated continuous risk score via a **FastAPI** microservice that maps predictions into actionable maintenance tiers:
  1. `NORMAL` ($p < \theta_1^*$): Standard continuous operation.
  2. `INSPECTION_REQUIRED` ($\theta_1^* \le p < \theta_2^*$): Triggers a low-cost preventive CBM work order (thermography/vibration check) at the next shift change.
  3. `CRITICAL_INTERVENTION` ($p \ge \theta_2^*$): Recommends immediate load derating and scheduled maintenance intervention.

---

## 2. Project Structure

```text
fleetreliability-ai/
├── .github/workflows/       # Automated CI pipeline (linting & unit tests)
├── data/
│   ├── raw/                 # Original UCI AI4I 2020 dataset
│   └── processed/           # Processed & split datasets
├── docs/
│   ├── PLAN.md              # Master Architecture & Execution Plan
│   └── course_guidelines/   # ML Zoomcamp rubrics & reference guides
├── notebooks/               # Exploratory Data Analysis (EDA) & Model Tuning
├── src/                     # Production source code (features, train, predict)
├── tests/                   # Pytest unit & API integration tests
├── Dockerfile               # Production container definition (python:3.11-slim)
├── docker-compose.yml       # Local container orchestration
├── pyproject.toml           # Poetry dependency & project configuration
├── poetry.lock              # Deterministic lockfile
├── requirements.txt         # Exported dependencies for universal reproducibility
└── README.md                # Project documentation
```

---

## 3. Environment Setup & Installation

This project uses **Python 3.11** and **Poetry** for deterministic dependency management (with an exported `requirements.txt` for universal compatibility).

### Option A: Using Poetry (Recommended for Development)
```bash
# 1. Clone the repository
git clone <REPO_URL>
cd fleetreliability-ai

# 2. Configure Poetry to create the virtualenv inside the project (.venv)
poetry config virtualenvs.in-project true

# 3. Install dependencies
poetry install --no-root

# 4. Activate the virtual environment (or run commands via `poetry run`)
source .venv/bin/activate
```

### Option B: Using Standard `venv` + `pip`
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 4. Running the Application & Docker

### Local Execution (once `src/train.py` and `src/predict.py` are implemented)
```bash
# Train the model and generate artifacts
poetry run python -m src.train

# Start the FastAPI inference service on port 9696
poetry run uvicorn src.predict:app --host 0.0.0.0 --port 9696 --reload
```

### Containerized Execution with Docker
```bash
# Build the image and start the container via Docker Compose
docker compose up --build -d

# Or using standard Docker CLI
docker build -t fleetreliability-ai .
docker run -it --rm -p 9696:9696 fleetreliability-ai
```

Interactive Swagger UI documentation will be available at:
- `http://localhost:9696/docs`
- `http://localhost:9696/health`

---

## 5. Architecture & Execution Roadmap

See [`docs/PLAN.md`](docs/PLAN.md) for the complete architecture diagram, physics-informed equations, Monte Carlo threshold formulation, and milestone schedule.
