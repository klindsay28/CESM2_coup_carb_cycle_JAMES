import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run_campaign_required",
        action="store_true",
        default=False,
        help="run campaign_required tests",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "campaign_required: mark test as needing campaign storage to run"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_campaign_required"):
        # --run_campaign_required given in cli: do not skip campaign_required tests
        return
    skip_campaign_required = pytest.mark.skip(
        reason="need --run_campaign_required option to run"
    )
    for item in items:
        if "campaign_required" in item.keywords:
            item.add_marker(skip_campaign_required)
