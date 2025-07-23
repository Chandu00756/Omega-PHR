# Contributing to Omega PHR

Thank you for your interest in contributing to the Omega PHR framework!
This guide will help you understand how to contribute effectively to this research-grade AI security testing platform.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Security Considerations](#security-considerations)
- [Research Ethics](#research-ethics)
- [Submission Process](#submission-process)

## Getting Started

Omega PHR is a sophisticated framework designed for AI security research with research-grade stability.
Before contributing, please:

1. Read our [Code of Conduct](CODE_OF_CONDUCT.md)
2. Review the project architecture and documentation
3. Set up your development environment
4. Start with smaller contributions before tackling major features

## Development Environment Setup

### Prerequisites

- Python 3.11 or higher
- Git
- Docker (optional, for containerized services)
- Access to appropriate testing environments

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/Chandu00756/Omega-PHR.git
cd omega-phr

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run initial tests
pytest tests/
```

### Environment Configuration

Create a `.env` file with appropriate configuration:

```env
OMEGA_DEBUG=true
OMEGA_LOG_LEVEL=DEBUG
OMEGA_MAX_AGENTS=10
TIMELINE_HOST=localhost
TIMELINE_PORT=50051
HIVE_HOST=localhost
HIVE_PORT=50052
```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

1. **Bug Reports**: Help us identify and fix issues
2. **Feature Requests**: Suggest new capabilities for the framework
3. **Code Contributions**: Implement new features or fix bugs
4. **Documentation**: Improve or expand documentation
5. **Research Papers**: Share findings using the framework
6. **Security Improvements**: Enhance security features
7. **Performance Optimizations**: Improve framework efficiency

### Before You Start

1. **Check existing issues**: Look for related issues or feature requests
2. **Discuss major changes**: Open an issue to discuss significant changes
3. **Fork the repository**: Create your own fork for development
4. **Create a branch**: Use descriptive branch names like `feature/agent-coordination` or `bugfix/timeline-sync`

## Code Standards

### Python Code Style

We follow PEP 8 with some specific guidelines:

- Line length: 100 characters maximum
- Use type hints for all function parameters and return values
- Document all classes and functions with docstrings
- Use meaningful variable and function names
- Follow async/await patterns for asynchronous code

### Code Quality Tools

We use several tools to maintain code quality:

```bash
# Code formatting
black omega_phr/ services/ tests/

# Import sorting
isort omega_phr/ services/ tests/

# Linting
ruff check omega_phr/ services/ tests/ --fix

# Type checking
mypy omega_phr/ --ignore-missing-imports

# Security scanning
bandit -r omega_phr/
```

### Example Code Structure

```python
"""
Module for advanced security testing operations.
Research-grade stability implementation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SecurityTest:
    """Represents a security test configuration."""

    id: str
    name: str
    target: str
    test_type: str
    parameters: Dict[str, Any]

    def validate(self) -> bool:
        """Validate test configuration."""
        if not self.id or not self.name:
            return False
        return True


class SecurityTestManager:
    """Manages security testing operations."""

    def __init__(self, config: SecurityConfig):
        """Initialize the security test manager."""
        self.config = config
        self.tests: Dict[str, SecurityTest] = {}

    async def run_test(self, test: SecurityTest) -> TestResult:
        """Execute a security test with research-grade stability."""
        try:
            logger.info(f"Starting security test: {test.name}")

            # Implementation here
            result = await self._execute_test(test)

            logger.info(f"Test completed: {test.name}")
            return result

        except Exception as e:
            logger.error(f"Test failed: {test.name}, error: {e}")
            raise

    async def _execute_test(self, test: SecurityTest) -> TestResult:
        """Internal test execution logic."""
        # Implementation details
        pass
```

## Testing Requirements

### Test Categories

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test interactions between components
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Validate performance requirements
5. **Security Tests**: Verify security features

### Writing Tests

```python
import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

class TestSecurityTestManager(unittest.TestCase):
    """Test security test management functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SecurityConfig()
        self.manager = SecurityTestManager(self.config)

    def test_test_validation(self):
        """Test security test validation."""
        valid_test = SecurityTest(
            id="test-1",
            name="Network Scan",
            target="192.168.1.0/24",
            test_type="network_scan",
            parameters={"ports": [80, 443]}
        )

        self.assertTrue(valid_test.validate())

    async def test_async_operations(self):
        """Test asynchronous operations."""
        test = SecurityTest(...)

        with patch.object(self.manager, '_execute_test') as mock_execute:
            mock_execute.return_value = TestResult(success=True)

            result = await self.manager.run_test(test)

            self.assertTrue(result.success)
            mock_execute.assert_called_once_with(test)
```

### Test Coverage

- Maintain minimum 80% code coverage
- All new features must include comprehensive tests
- Critical security functions require 95%+ coverage
- Include both positive and negative test cases

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=omega_phr --cov-report=html

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v

# Run performance tests
pytest tests/performance/ -v --benchmark-only
```

## Documentation

### Documentation Standards

- Use clear, concise language
- Provide practical examples
- Include both conceptual and reference documentation
- Update documentation with code changes
- Follow Markdown standards for formatting

### Types of Documentation

1. **API Documentation**: Automatically generated from docstrings
2. **User Guides**: Step-by-step instructions for common tasks
3. **Architecture Documentation**: System design and component interactions
4. **Research Papers**: Academic documentation of findings
5. **Security Advisories**: Important security information

### Documentation Structure

``
docs/
├── api/                    # API reference documentation
├── guides/                 # User and developer guides
├── architecture/           # System architecture documentation
├── research/              # Research papers and findings
├── security/              # Security documentation
└── examples/              # Code examples and tutorials
``

## Security Considerations

### Secure Development Practices

1. **Input Validation**: Validate all external inputs
2. **Authentication**: Implement proper authentication mechanisms
3. **Authorization**: Enforce access controls consistently
4. **Encryption**: Use encryption for sensitive data
5. **Logging**: Log security events appropriately
6. **Error Handling**: Avoid exposing sensitive information in errors

### Security Review Process

All contributions undergo security review:

1. **Automated Security Scanning**: Using bandit and safety
2. **Code Review**: Manual review by security-aware developers
3. **Threat Modeling**: Analysis of potential attack vectors
4. **Penetration Testing**: Testing of security features
5. **Third-party Dependencies**: Regular security audits

### Vulnerability Reporting

If you discover a security vulnerability:

1. **Do NOT** create a public issue
2. Email security details to [security@chandu00756.dev]
3. Allow time for investigation and fix
4. Follow responsible disclosure practices

## Research Ethics

### Ethical Guidelines

1. **Authorized Testing Only**: Only test systems you own or have explicit permission to test
2. **Responsible Disclosure**: Follow established vulnerability disclosure timelines
3. **Data Privacy**: Protect any sensitive data encountered during research
4. **Academic Integrity**: Properly cite and attribute research contributions
5. **Harm Minimization**: Ensure research activities do not cause harm

### Research Documentation

When contributing research:

- Document methodology clearly
- Provide reproducible results
- Include limitations and future work
- Follow academic citation standards
- Share datasets responsibly (when possible)

## Submission Process

### Pull Request Guidelines

1. **Branch Naming**: Use descriptive names like `feature/agent-coordination`
2. **Commit Messages**: Follow conventional commits format
3. **Small Changes**: Keep pull requests focused and reasonably sized
4. **Testing**: Include appropriate tests for your changes
5. **Documentation**: Update documentation as needed

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Security improvement

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Security review completed

## Documentation
- [ ] Code is documented
- [ ] User documentation updated
- [ ] API documentation updated

## Security
- [ ] No sensitive data exposed
- [ ] Input validation implemented
- [ ] Authentication/authorization considered
- [ ] Security scanning passed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Appropriate tests added
- [ ] Documentation updated
- [ ] No breaking changes (or properly documented)
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs automated tests
2. **Code Review**: At least one maintainer reviews the code
3. **Security Review**: Security-focused review for sensitive changes
4. **Documentation Review**: Verify documentation is complete and accurate
5. **Integration Testing**: Test integration with existing components

### Merge Requirements

- All automated checks must pass
- At least one approving review from a maintainer
- Security review approval (for security-related changes)
- Documentation updates completed
- No unresolved review comments

## Getting Help

### Resources

- **Documentation**: Check the docs/ directory
- **Issues**: Search existing issues for similar problems
- **Discussions**: Use GitHub Discussions for questions
- **Chat**: Join our research community chat
- **Email**: Contact maintainers directly for sensitive issues

### Community Guidelines

- Be respectful and professional
- Provide clear and detailed information
- Search existing resources before asking questions
- Help others in the community when possible
- Follow the Code of Conduct in all interactions

## Recognition

We value all contributions to the Omega PHR framework:

- **Contributors**: Listed in CONTRIBUTORS.md
- **Research Citations**: Academic papers citing the framework
- **Security Acknowledgments**: Security researchers who report vulnerabilities
- **Community Leaders**: Active community members and maintainers

Thank you for contributing to advancing AI security research!
