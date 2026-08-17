"""Run the detection command line with ``python -m neutral_atom_mht``.

The module contains no project logic.  It simply forwards terminal arguments
to :func:`neutral_atom_mht.cli.main` and returns that function's exit code.
"""

from .cli import main

raise SystemExit(main())
