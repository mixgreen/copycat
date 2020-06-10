import matplotlib  # type: ignore

# Set the Matplotlib backend
# Workaround to fix QT error with Matplotlib, see https://nixos.wiki/wiki/Qt
matplotlib.use('pdf')
