def pytest_configure(config):
    config.addinivalue_line("markers", "slow: resource-budget checks; opt-in")
