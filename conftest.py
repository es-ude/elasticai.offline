import os

from hypothesis import HealthCheck, settings

pytest_plugins = ["elasticai.creator.testing"]

# Deterministic in CI (same examples every run), randomized locally.
_common = dict(
    deadline=None,
    max_examples=10,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
settings.register_profile("ci", derandomize=True, **_common)
settings.register_profile("dev", **_common)
settings.load_profile("ci" if os.environ.get("CI") else "dev")


def pytest_sessionstart(session):
    from elasticai.offline import get_path_to_project
    from shutil import rmtree

    rmtree(get_path_to_project("config"), ignore_errors=True)
    rmtree(get_path_to_project("dataset"), ignore_errors=True)
    rmtree(get_path_to_project("runs"), ignore_errors=True)
    rmtree(get_path_to_project("src_dnn"), ignore_errors=True)
    rmtree(get_path_to_project("temp_test"), ignore_errors=True)


def pytest_sessionfinish(session, exitstatus):
    from elasticai.offline import get_path_to_project
    from shutil import rmtree

    rmtree(get_path_to_project("config"), ignore_errors=True)
    rmtree(get_path_to_project("dataset"), ignore_errors=True)
    rmtree(get_path_to_project("src_dnn"), ignore_errors=True)
    rmtree(get_path_to_project("temp_test"), ignore_errors=True)
