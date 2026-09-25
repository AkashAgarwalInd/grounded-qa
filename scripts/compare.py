import json
import sys
import pathlib


def main():
    if len(sys.argv) < 3:
        print("Usage: scripts/compare.py <hashA> <hashB>", file=sys.stderr)
        sys.exit(1)

    a, b = (json.loads((pathlib.Path("runs") / h / "config.json").read_text())
            for h in sys.argv[1:3])

    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            print(f'{k:16} {str(a.get(k)):>24}  ->  {str(b.get(k))}')


if __name__ == "__main__":
    main()