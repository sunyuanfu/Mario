from __future__ import annotations

from mario_llm_stage2.cli import parse_args


def main() -> None:
    args = parse_args()
    from mario_llm_stage2.pipeline import run_stage2

    run_stage2(args)


if __name__ == "__main__":
    main()
