import pytest


def pytest_addoption(parser):
    parser.addoption(
        '--slow', action='store_true', default=False, help='run slow tests'
    )


def pytest_collection_modifyitems(config, items):
    # skip test marked as slow except if --slow option passed to the CLI
    if config.getoption('--slow'):
        return
    skip_slow = pytest.mark.skip(reason='need --slow option to run')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip_slow)
