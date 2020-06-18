"""Force the Matplotlib backend to PDF

Workaround to fix QT error with Matplotlib, see https://nixos.wiki/wiki/Qt.

In general this workaround is not required when using the ARTIQ dashboard,
but it is often needed when using Matplotlib without ARTIQ in the ARTIQ Nix environment.
"""

import matplotlib  # type: ignore

matplotlib.use('pdf')
