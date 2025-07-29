Contributing
============

We welcome contributions to Omega-PHR! This guide will help you get started with contributing to the project.

Development Environment
-----------------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/yourusername/Omega-PHR.git
      cd Omega-PHR

3. Install in development mode:

   .. code-block:: bash

      pip install -e .
      pip install pytest pytest-asyncio pytest-cov black ruff mypy pre-commit

4. Set up pre-commit hooks:

   .. code-block:: bash

      pre-commit install

Code Standards
--------------

We maintain high code quality standards:

Formatting:
~~~~~~~~~~~

* **Black**: Code formatting
* **isort**: Import sorting
* **Ruff**: Linting and code quality

Type Checking:
~~~~~~~~~~~~~~

* **MyPy**: Static type checking
* All new code should include type hints

Testing:
~~~~~~~~

* **pytest**: Testing framework
* Aim for high test coverage
* Include both unit and integration tests

Running Quality Checks
----------------------

Before submitting changes:

1. Format code:

   .. code-block:: bash

      black omega_phr/ services/ tests/

2. Check linting:

   .. code-block:: bash

      ruff check omega_phr/ services/ tests/ --fix

3. Run type checking:

   .. code-block:: bash

      mypy omega_phr/ --ignore-missing-imports

4. Run tests:

   .. code-block:: bash

      pytest tests/ -v --cov=omega_phr

5. Run pre-commit checks:

   .. code-block:: bash

      pre-commit run --all-files

Submitting Changes
------------------

1. Create a feature branch:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. Make your changes and ensure all tests pass
3. Commit your changes with descriptive messages:

   .. code-block:: bash

      git commit -m "Add feature: description of your changes"

4. Push to your fork:

   .. code-block:: bash

      git push origin feature/your-feature-name

5. Create a Pull Request on GitHub

Pull Request Guidelines
-----------------------

* Provide a clear description of changes
* Include tests for new functionality
* Ensure all CI checks pass
* Update documentation as needed
* Follow the existing code style

Issue Reporting
---------------

When reporting issues:

* Use a clear and descriptive title
* Provide steps to reproduce the issue
* Include relevant system information
* Attach logs or error messages if applicable

Development Workflow
--------------------

1. **Planning**: Discuss major changes in issues first
2. **Development**: Follow coding standards and write tests
3. **Testing**: Ensure comprehensive test coverage
4. **Documentation**: Update docs for new features
5. **Review**: Submit PR for code review
6. **Integration**: Merge after approval and CI success

Areas for Contribution
----------------------

* **Bug fixes**: Address reported issues
* **Feature development**: Implement new capabilities
* **Documentation**: Improve guides and API docs
* **Testing**: Increase test coverage
* **Performance**: Optimize system performance
* **Security**: Enhance system security

Getting Help
------------

* **Issues**: GitHub issue tracker
* **Discussions**: GitHub discussions
* **Code Review**: Pull request comments

Thank you for contributing to Omega-PHR!
