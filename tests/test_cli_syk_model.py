import sys

from quench_protocols.cli import _parse_args


def test_cli_parses_syk_model_args(monkeypatch) -> None:
    argv = [
        "prog",
        "--model",
        "syk",
        "--protocol",
        "two",
        "--n-fermions",
        "6",
        "--J",
        "1.0",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = _parse_args()
    assert args.model == "syk"
    assert args.n_fermions == 6
    assert args.J == 1.0
