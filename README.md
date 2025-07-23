# Omega-Paradox Hive Recursion (Ω-PHR)

## Revolutionary AI Security Testing Framework

The Omega-Paradox Hive Recursion (Ω-PHR) project represents a groundbreaking advancement in AI security testing,
introducing the world's first comprehensive framework that simultaneously integrates temporal paradox testing,
synthetic adversarial hive attacks, recursive memory inversion, and generative infinite loop fuzzing.

## Architecture Overview

### Core Components

1. **Timeline Lattice Service** - Temporal paradox simulation and causality testing
2. **Hive Orchestrator** - Autonomous adversarial swarm coordination
3. **Memory Inversion Engine** - Recursive memory manipulation and rollback
4. **Recursive Loop Synthesizer** - Generative infinite loop detection and containment
5. **Ω-State Register** - Entropy quarantine and anomaly containment
6. **Telemetry Exporter** - Real-time monitoring and metrics collection

### Key Features

- **Multi-dimensional stress testing** across temporal, hive, memory, and recursive axes
- **Entropy-based containment** for runaway processes and paradox states
- **Real-time monitoring** with Prometheus integration
- **Scalable architecture** using Ray for distributed processing
- **Production-ready** with Docker containerization and Kubernetes deployment

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- 8GB+ RAM (16GB recommended)
- CUDA-compatible GPU (optional, for research workloads)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Chandu00756/Omega-PHR.git
cd omega-phr
```

2.Set up the development environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .[dev]
```

3.Install pre-commit hooks:

```bash
pre-commit install
```

4.Generate Protocol Buffer stubs:

```bash
make proto
```

### Running Locally

#### Option 1: Individual Services (Development)

```bash
# Terminal 1 - Timeline Lattice Service
cd services/timeline_lattice
python main.py

# Terminal 2 - Hive Orchestrator
cd services/hive_orchestrator
python main.py

# Terminal 3 - Memory Inversion Service
cd services/memory_inversion
python main.py

# Terminal 4 - Recursive Loop Synthesizer
cd services/recursive_loop_synth
python main.py

# Terminal 5 - Telemetry Exporter
cd services/telemetry_exporter
python main.py
```

#### Option 2: Docker Compose (Integration Testing)

```bash
docker-compose up --build
```

### Running Tests

```bash
# Unit tests
pytest tests/unit -m unit

# Integration tests
pytest tests/integration -m integration

# End-to-end tests
pytest tests/e2e -m e2e

# All tests with coverage
pytest --cov=omega_phr --cov-report=html
```

## Usage Examples

### Basic Temporal Paradox Testing

```python
from omega_phr.timeline import TimelineLattice
from omega_phr.models import Event

# Initialize timeline lattice
lattice = TimelineLattice()

# Create temporal paradox scenario
event = Event(
    event_id="test-001",
    actor_id="ai-agent",
    timeline_id="alpha",
    payload={"prompt": "What year is it?"},
    valid_at_us=1234567890000000
)

# Test temporal consistency
result = lattice.test_paradox(event)
print(f"Paradox detected: {result.has_paradox}")
```

### Hive Attack Orchestration

```python
from omega_phr.hive import HiveOrchestrator
from omega_phr.attackers import InjectionAttacker, LogicBombAttacker

# Initialize hive with multiple attack strategies
hive = HiveOrchestrator()
hive.add_attacker(InjectionAttacker(persona="social_engineer"))
hive.add_attacker(LogicBombAttacker(persona="recursive_bomber"))

# Execute coordinated attack
results = await hive.coordinate_attack(target_model="gpt-4", scenario="jailbreak")
print(f"Attack success rate: {results.success_rate}")
```

### Memory Inversion Testing

```python
from omega_phr.memory import MemoryInverter

# Initialize memory inverter
inverter = MemoryInverter()

# Test memory rollback scenario
original_state = {"context": "I am a helpful AI assistant"}
inverted = inverter.invert_memory(original_state, strategy="contradiction")

print(f"Original: {original_state}")
print(f"Inverted: {inverted}")
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCYLLA_HOSTS` | `127.0.0.1` | Cassandra/ScyllaDB hosts |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection string |
| `RAY_HEAD_NODE` | `ray://localhost:10001` | Ray cluster head node |
| `PROMETHEUS_PORT` | `9090` | Prometheus metrics port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `OMEGA_STATE_THRESHOLD` | `0.1` | Entropy threshold for Ω-state detection |

### Service Configuration Files

- `services/timeline_lattice/config.yaml` - Timeline service configuration
- `services/hive_orchestrator/config.yaml` - Hive orchestrator settings
- `services/memory_inversion/config.yaml` - Memory inversion parameters
- `services/recursive_loop_synth/config.yaml` - Loop synthesis configuration

## Architecture Deep Dive

### Timeline Lattice

The Timeline Lattice implements a novel approach to temporal consistency testing using:

- **Layered Temporal Realities (LTR)**: Multiple timeline branches with causal validation
- **Event Sourcing**: Immutable event log with cryptographic signatures
- **Temporal Paradox Detection**: Real-time contradiction analysis
- **Timeline Merge/Fork Operations**: Dynamic timeline manipulation

### Hive Orchestrator

The Hive Orchestrator coordinates multiple autonomous agents using:

- **Ray Actors**: Distributed agent execution across cluster nodes
- **Attack Strategy Patterns**: Pluggable attack methodologies
- **Swarm Intelligence**: Emergent coordination behaviors
- **Real-time Communication**: Actor message passing and state synchronization

### Memory Inversion Engine

The Memory Inversion Engine provides:

- **Reversible Operations**: Rollback capabilities for all memory modifications
- **Inversion Strategies**: Multiple approaches to memory manipulation
- **Consistency Validation**: Detection of memory-based contradictions
- **Forensic Logging**: Complete audit trail of all operations

### Recursive Loop Synthesizer

The Loop Synthesizer implements:

- **Generative Loop Creation**: AI-driven generation of recursive scenarios
- **Entropy Monitoring**: Real-time entropy analysis for loop detection
- **Containment Protocols**: Automatic quarantine of runaway processes
- **Loop Classification**: Taxonomy of different loop types and behaviors

## Monitoring and Observability

### Metrics

The framework exposes comprehensive metrics via Prometheus:

- Timeline consistency scores
- Hive attack success rates
- Memory inversion statistics
- Loop detection and containment metrics
- System resource utilization

### Logging

Structured logging using `structlog` with:

- JSON output format for machine parsing
- Correlation IDs for request tracing
- Sensitive data redaction
- Multiple log levels and filtering

### Dashboards

Pre-built Grafana dashboards for:

- Real-time system overview
- Attack campaign analysis
- Performance monitoring
- Alert management

## Security Considerations

### Containment Protocols

- **Entropy Quarantine**: Automatic isolation of high-entropy states
- **Process Sandboxing**: Containerized execution environments
- **Network Isolation**: Restricted inter-service communication
- **Resource Limits**: CPU, memory, and time constraints

### Data Protection

- **Encryption at Rest**: All persistent data encrypted
- **Secure Communication**: TLS for all inter-service communication
- **Access Control**: Role-based permissions and API keys
- **Audit Logging**: Complete audit trail for compliance

## Performance and Scalability

### Horizontal Scaling

- **Microservices Architecture**: Independent service scaling
- **Ray Cluster**: Distributed processing across multiple nodes
- **Database Sharding**: Partitioned data storage
- **Load Balancing**: Intelligent request distribution

### Performance Optimizations

- **Async I/O**: Non-blocking operations throughout
- **Connection Pooling**: Efficient resource utilization
- **Caching Strategies**: Redis-based caching layers
- **Batch Processing**: Optimized bulk operations

## Deployment

### Local Development

```bash
# Start all services
docker-compose up -d

# Scale specific services
docker-compose up --scale hive_orchestrator=3
```

### Production (Kubernetes)

```bash
# Deploy to Kubernetes cluster
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=omega-phr
```

### Google Cloud Platform

```bash
# Deploy using Cloud Build
gcloud builds submit --config cloudbuild.yaml

# Deploy to GKE
gcloud container clusters get-credentials omega-phr-cluster
kubectl apply -f k8s/gcp/
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and add tests
4. Run the test suite: `pytest`
5. Commit changes: `git commit -am 'Add my feature'`
6. Push to branch: `git push origin feature/my-feature`
7. Submit a pull request

### Code Quality Standards

- **Type Hints**: All public APIs must include type annotations
- **Documentation**: Docstrings required for all public functions/classes
- **Testing**: Minimum 90% test coverage for new code
- **Linting**: Code must pass `ruff` and `black` formatting
- **Security**: All dependencies scanned for vulnerabilities

## Research and Publications

### Academic Impact

This framework has been designed to support cutting-edge research in:

- AI safety and robustness testing
- Adversarial machine learning
- Temporal reasoning in AI systems
- Multi-agent system security
- Emergent behavior analysis

### Citation

If you use Ω-PHR in your research, please cite:

```bibtex
@software{chandu2025omegaphr,
  title={Omega-Paradox Hive Recursion: Revolutionary AI Security Testing Framework},
  author={Chitikam, Venkata Sai Chandu},
  year={2025},
  url={https://github.com/Chandu00756/Omega-PHR},
  version={1.0.0}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support and Community

- **Documentation**: [https://omega-phr.readthedocs.io](https://omega-phr.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/Chandu00756/Omega-PHR/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Chandu00756/Omega-PHR/discussions)
- **Discord**: [Ω-PHR Community](https://discord.gg/omega-phr)

## Acknowledgments

- MITRE ATLAS for threat modeling frameworks
- NIST AI RMF for governance guidelines
- Ray team for distributed computing platform
- ScyllaDB team for high-performance database engine

---

**Ω-PHR**: Where temporal paradoxes meet adversarial swarms in the quantum realm of AI security testing.
