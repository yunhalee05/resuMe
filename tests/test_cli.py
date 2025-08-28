from resume.cli import main


def test_greet_world(capsys):
    exit_code = main(["greet"])  # default World
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Hello, World!" in captured.out


def test_greet_name(capsys):
    exit_code = main(["greet", "Yuna"])  # specific name
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Hello, Yuna!" in captured.out


