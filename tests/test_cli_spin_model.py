import sys

from quench_protocols.cli import _parse_args


def test_cli_parses_spin_model_args(monkeypatch) -> None:
    argv = [
        "prog",
        "--model",
        "spin",
        "--protocol",
        "two",
        "--n-spins",
        "4",
        "--J",
        "1.0",
        "--xi-x",
        "1",
        "--xi-y",
        "1",
        "--xi-z",
        "-2",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = _parse_args()
    assert args.model == "spin"
    assert args.n_spins == 4
    assert args.J == 1.0
    assert (args.xi_x, args.xi_y, args.xi_z) == (1.0, 1.0, -2.0)
