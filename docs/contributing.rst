Contributing
============

We welcome contributions to idx-flow! This document provides guidelines for contributing.

Getting Started
---------------

1. Fork the repository on GitHub: https://github.com/otaviomf123/idx-flow
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/idx-flow.git
      cd idx-flow

3. Create a virtual environment and install dependencies:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      pip install -e ".[dev]"

4. Create a branch for your changes:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

Development Workflow
--------------------

Running Tests
^^^^^^^^^^^^^

.. code-block:: bash

   pytest tests/ -v

Running the verification script:

.. code-block:: bash

   python verify_build.py

Code Formatting
^^^^^^^^^^^^^^^

We use black and isort for code formatting:

.. code-block:: bash

   black src/ tests/
   isort src/ tests/

Type Checking
^^^^^^^^^^^^^

.. code-block:: bash

   mypy src/idx_flow/

Building Documentation
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd docs
   make html

The documentation will be in ``docs/_build/html/``.

Coding Standards
----------------

- Follow PEP 8 style guidelines
- Use type hints for function arguments and return values
- Write Google-style docstrings for all public functions and classes
- Keep lines under 100 characters
- Write tests for new functionality

Docstring Example
^^^^^^^^^^^^^^^^^

.. code-block:: python

   def my_function(arg1: int, arg2: str = "default") -> bool:
       """
       Short description of the function.

       Longer description if needed, explaining the function's
       purpose and behavior.

       Args:
           arg1: Description of arg1.
           arg2: Description of arg2. Default is "default".

       Returns:
           Description of the return value.

       Raises:
           ValueError: When arg1 is negative.

       Example:
           >>> result = my_function(42, "hello")
           >>> print(result)
           True
       """
       pass

Pull Request Process
--------------------

1. Ensure all tests pass
2. Update documentation if needed
3. Add an entry to CHANGELOG if appropriate
4. Submit a pull request with a clear description

Your PR should:

- Have a clear, descriptive title
- Reference any related issues
- Include tests for new functionality
- Pass all CI checks

Reporting Issues
----------------

When reporting issues, please include:

- Python version
- PyTorch version
- Operating system
- Minimal code to reproduce the issue
- Full error traceback

Questions?
----------

Feel free to open an issue for questions or discussions about the project.

Thank you for contributing!
