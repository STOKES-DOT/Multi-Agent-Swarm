def test_cli_version(capsys) -> None:
    from multi_agent_pso.cli import main

    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "0.1.0"
