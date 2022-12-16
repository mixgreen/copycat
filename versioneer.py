import os
import sys

LONG_VERSION_PY = r'''
def get_version():
    return "{version}"
'''


def write_to_version_file(filename, version):
    """Write the given version number to the given _version.py file."""
    os.unlink(filename)
    with open(filename, "w") as f:
        f.write(LONG_VERSION_PY.format(version=version))

    print("set %s to '%s'" % (filename, version))


def get_version():
    return os.getenv("VERSIONEER_OVERRIDE", default="7.0.unknown")


def get_cmdclass():
    cmds = {}

    # we override different "build_py" commands for both environments
    if "setuptools" in sys.modules:
        from setuptools.command.build_py import build_py as _build_py
    else:
        from distutils.command.build_py import build_py as _build_py

    class cmd_build_py(_build_py):
        def run(self):
            version = get_version()
            _build_py.run(self)
            target_versionfile = os.path.join(self.build_lib, "dax", "_version.py")
            print("UPDATING %s" % target_versionfile)
            write_to_version_file(target_versionfile, version)

    cmds["build_py"] = cmd_build_py

    if "setuptools" in sys.modules:
        from setuptools.command.sdist import sdist as _sdist
    else:
        from distutils.command.sdist import sdist as _sdist

    class cmd_sdist(_sdist):
        def run(self):
            version = get_version()
            self._versioneer_generated_version = version
            # unless we update this, the command will keep using the old
            # version
            self.distribution.metadata.version = version
            return _sdist.run(self)

        def make_release_tree(self, base_dir, files):
            _sdist.make_release_tree(self, base_dir, files)
            # now locate _version.py in the new base_dir directory
            # (remembering that it may be a hardlink) and replace it with an
            # updated value
            target_versionfile = os.path.join(base_dir, "dax", "_version.py")
            print("UPDATING %s" % target_versionfile)
            write_to_version_file(target_versionfile,
                                  self._versioneer_generated_version)

    cmds["sdist"] = cmd_sdist

    return cmds
