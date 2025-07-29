Services
========

Omega-PHR Architecture
----------------------

The Omega-PHR system is built on a microservices architecture where each service handles specific functionality. This design provides scalability, maintainability, and fault tolerance.

Timeline Lattice Service
------------------------

**Location**: ``services/timeline_lattice/``

The Timeline Lattice service manages temporal relationships and chronological data organization within the PHR system.

Features:
~~~~~~~~~

* Temporal data indexing and retrieval
* Timeline-based event correlation
* Historical data analysis
* Time-series data processing

Configuration:
~~~~~~~~~~~~~~

* **Host**: ``localhost`` (configurable via ``TIMELINE_HOST``)
* **Port**: ``50051`` (configurable via ``TIMELINE_PORT``)
* **Protocol**: gRPC

Hive Orchestrator Service
-------------------------

**Location**: ``services/hive_orchestrator/``

The Hive Orchestrator coordinates inter-service communication and manages workflow orchestration across the entire system.

Features:
~~~~~~~~~

* Service discovery and registration
* Workflow coordination
* Load balancing
* Health monitoring

Configuration:
~~~~~~~~~~~~~~

* **Host**: ``localhost`` (configurable via ``HIVE_HOST``)
* **Port**: ``50052`` (configurable via ``HIVE_PORT``)
* **Protocol**: gRPC

Memory Inversion Service
-----------------------

**Location**: ``services/memory_inversion/``

Provides intelligent data processing and pattern recognition capabilities for complex healthcare data analysis.

Features:
~~~~~~~~~

* Pattern recognition algorithms
* Data transformation pipelines
* Machine learning integration
* Anomaly detection

Recursive Loop Synthesis Service
-------------------------------

**Location**: ``services/recursive_loop_synth/``

Analyzes complex data relationships and dependencies within the healthcare data ecosystem.

Features:
~~~~~~~~~

* Dependency analysis
* Relationship mapping
* Cyclic pattern detection
* Data flow optimization

Telemetry Exporter Service
--------------------------

**Location**: ``services/telemetry_exporter/``

Exports system metrics and provides comprehensive monitoring capabilities for the entire Omega-PHR system.

Features:
~~~~~~~~~

* Metrics collection and aggregation
* Performance monitoring
* Error tracking and alerting
* System health dashboards

Service Communication
--------------------

Services communicate using:

* **gRPC**: For high-performance inter-service communication
* **Protocol Buffers**: For efficient data serialization
* **Structured Logging**: For comprehensive system observability

Running Services
----------------

Individual Services:
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Timeline Lattice
   python services/timeline_lattice/main.py

   # Hive Orchestrator
   python services/hive_orchestrator/main.py

   # Memory Inversion
   python services/memory_inversion/main.py

Using VS Code Tasks:
~~~~~~~~~~~~~~~~~~~

The project includes pre-configured VS Code tasks for easy service management:

* ``Start Timeline Service``
* ``Start Hive Service``
* ``Docker Compose Up``

Docker Deployment:
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker-compose up -d

Service Health Checks
---------------------

Each service provides health check endpoints for monitoring and orchestration. The system includes comprehensive logging and monitoring capabilities to ensure reliable operation.
