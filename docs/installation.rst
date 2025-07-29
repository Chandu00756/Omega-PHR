Installation
============

Requirements
------------

* Python 3.8 or higher
* pip package manager
* Git

Quick Installation
------------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/Chandu00756/Omega-PHR.git
      cd Omega-PHR

2. Install the package in development mode:

   .. code-block:: bash

      pip install -e .

3. Install development dependencies:

   .. code-block:: bash

      pip install pytest pytest-asyncio pytest-cov black ruff mypy pre-commit

Development Setup
-----------------

For development work, additional setup is recommended:

1. Install pre-commit hooks:

   .. code-block:: bash

      pre-commit install

2. Run the test suite:

   .. code-block:: bash

      pytest tests/ -v

3. Format code:

   .. code-block:: bash

      black omega_phr/ services/ tests/

4. Lint code:

   .. code-block:: bash

      ruff check omega_phr/ services/ tests/ --fix

Docker Installation
-------------------

You can also run Omega-PHR using Docker:

1. Build the Docker image:

   .. code-block:: bash

      docker build -t omega-phr:latest .

2. Run with Docker Compose:

   .. code-block:: bash

      docker-compose up -d

Verification
------------

To verify your installation:

.. code-block:: bash

   python -c "import omega_phr; print('Omega-PHR installed successfully!')"

Troubleshooting
---------------

Common installation issues:

* **Python version**: Ensure you're using Python 3.8+
* **Virtual environment**: Consider using a virtual environment for isolated installation
* **Dependencies**: Make sure all required packages are installed
