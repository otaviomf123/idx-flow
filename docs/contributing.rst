Contributing
============

Contributions are welcome.

Setup
-----

1. Fork the repo: https://github.com/otaviomf123/idx-flow
2. Clone and install:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/idx-flow.git
      cd idx-flow
      python -m venv venv
      source venv/bin/activate
      pip install -e ".[dev]"

3. Create a branch:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

Running Tests
-------------

.. code-block:: bash

   pytest tests/ -v

Formatting
----------

.. code-block:: bash

   black src/ tests/
   isort src/ tests/

Type Checking
-------------

.. code-block:: bash

   mypy src/idx_flow/

Building Docs
-------------

.. code-block:: bash

   cd docs
   make html

Output goes to ``docs/_build/html/``.

Code Style
----------

- PEP 8
- Type hints on public functions
- Google-style docstrings
- Lines under 100 characters
- Tests for new functionality

Pull Requests
-------------

1. Make sure tests pass.
2. Update docs if needed.
3. Add a changelog entry if appropriate.
4. Open a PR with a clear description.

Reporting Issues
----------------

Include: Python version, PyTorch version, OS, minimal reproducer, full traceback.
