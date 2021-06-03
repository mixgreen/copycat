"""
Force the Matplotlib backend to PDF.

Importing this module activates a workaround for the matplotlib backend.
This workaround is required to prevent QT related errors, see https://nixos.wiki/wiki/Qt.
Alternatively, users can also set the following environment variable: ``MPLBACKEND=pdf``.

In general this workaround is not required when using the ARTIQ dashboard or other ``artiq_*`` commands.
This workaround is sometimes needed when using Matplotlib in the ARTIQ Nix environment without using the
``artiq_*`` commands, for example during testing or debugging.
"""

import matplotlib

matplotlib.use('pdf')
