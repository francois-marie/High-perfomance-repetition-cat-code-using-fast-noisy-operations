from subprocess import CalledProcessError, run


def _init_version():
    """Return a string identifying the git version of qsim.

    The returned string is the output of `git rev-parse HEAD`, ie the hash of
    the commit to which the HEAD currently points.
    """

    try:
        result = run(
            ['git', 'rev-parse', 'HEAD'],
            check=True,
            shell=False,
            capture_output=True,
        )
        return result.stdout.decode().strip()
    except (FileNotFoundError, CalledProcessError):
        pass

    return 'UNKNOWN'


VERSION = _init_version()
"""A string identifying the git version of qsim. Warning: this variable is not
updated by the `autoreload` extension for Jupyter when adding a new commit."""
