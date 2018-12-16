import pandas as pd
import sys

def main(args):
    f = pd.read_csv(args[0])
    i = pd.read_csv(args[1])
    f = f.dropna(axis=1)
    merged = i.merge(f, on=args[2])
    merged.to_csv(args[3], index=False)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))