"""`python -m pipeline ...` entrypoint.

Forwards to pipeline.cli.main so users can drop the `.cli` segment.
"""

import sys

from .cli import main


if __name__ == "__main__":
    sys.exit(main())
