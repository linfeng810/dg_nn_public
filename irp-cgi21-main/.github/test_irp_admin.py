import json
import pytest
import pathlib

EIGHT_MB = 8_388_608  # in bytes

reports = pathlib.Path('./reports')
info_file = pathlib.Path('./info/info.json')


@pytest.fixture(scope='session')
def info():
    """Retrieve info.json content as dict."""
    with open(info_file, 'r', encoding='utf-8') as fin:
        yield json.load(fin)


@pytest.fixture(scope='session')
def username(info):
    """Extract username from info.json."""
    return info['username']


@pytest.fixture(scope='session')
def project_plan(username):
    """Derive project plan path."""
    return reports / f'{username}-project-plan.pdf'


@pytest.fixture(scope='session')
def final_report(username):
    """Derive final report path."""
    return reports / f'{username}-final-report.pdf'


@pytest.fixture(scope='session')
def presentation(username):
    """Derive presentation path."""
    return reports / f'{username}-presentation.pdf'


def check_supervisor(supervisor):
    """Check supervisor data.
    
    Helper function to reduce coderep.

    """
    assert isinstance(supervisor, dict)
    assert supervisor  # assert it is not empty
    for field in ['name', 'email', 'affiliation']:
        assert isinstance(supervisor[field], str)
        assert supervisor[field]

    return True


class TestInfo:
    """Tests grouped for info.json."""
    def test_info_username(self, username):
        assert isinstance(username, str)
        assert username  # not empty

    def test_info_title(self, info):
        title = info['title']
        assert isinstance(title, str)
        assert title

    def test_info_s1(self, info):
        assert check_supervisor(info['s1'])

    def test_info_s2(self, info):
        assert check_supervisor(info['s2'])


class TestProjectPlan:
    """Tests grouped for project plan."""
    def test_name(self, project_plan):
        assert project_plan.is_file()
        assert str(project_plan).islower()

    def test_size(self, project_plan):
        print(project_plan.stat().st_size)
        assert project_plan.stat().st_size <= EIGHT_MB


class TestFinalReport:
    """Tests grouped for final report."""
    def test_name(self, final_report):
        assert final_report.is_file()
        assert str(final_report).islower()

    def test_size(self, final_report):
        print(final_report.stat().st_size)
        assert final_report.stat().st_size <= EIGHT_MB


class TestPresentation:
    """Tests grouped for presentation."""
    def test_name(self, presentation):
        assert presentation.is_file()
        assert str(presentation).islower()

    def test_size(self, presentation):
        print(presentation.stat().st_size)
        assert presentation.stat().st_size <= EIGHT_MB
