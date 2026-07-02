import cafein


def test_version_is_exposed():
    major, minor, patch = cafein.__version__.split(".")
    assert major.isdigit()
    assert minor.isdigit()
