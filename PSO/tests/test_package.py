def test_package_exposes_version() -> None:
    import multi_agent_pso

    assert multi_agent_pso.__version__ == "0.1.0"
