import argparse

from sca import SCA

# todo: add naming of grouping
parser = argparse.ArgumentParser(description="Calculate collocates")
# parser.add_argument("name", "")
parser.add_argument(
    "--collocates",
    "-c",
    help="Collocate patterns as string, using comma as a separator between collocates",
)
parser.add_argument(
    "--window",
    "-w",
    help="Window size to be used if not specified in the 'collocates' argument, defaults to 10",
    default=10,
)
parser.add_argument(
    "--output", "-o", help="Output file, .tsv format", default="output.tsv"
)
args = parser.parse_args()

if __name__ == "__main__":
    cols = []
    for collocate in args.collocates.split(","):
        pattern1, pattern2, window = collocate.strip().split()
        pattern1, pattern2 = sorted((pattern1, pattern2))
        window = int(window)

        cols.append((pattern1, pattern2, window))

    patterns1, patterns2, windows = zip(*cols)

    sca = SCA()

    sca.add_collocates(zip(patterns1, patterns2))

    # todo: add output file.

    sca.counts_by_subgroups(cols, args.output)
