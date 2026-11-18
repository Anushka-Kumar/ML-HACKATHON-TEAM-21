
import os

SAMPLE_TRIPS = [
    ("train", "00000003"),
    ("test", "00000000")
]

EXTENSIONS = [
    ".track",
    ".nodes",
    ".arcs",
    ".route"
]


def inspect_file(partition, trip_id, extension):

    filepath = os.path.join(
        "data",
        partition,
        trip_id,
        trip_id + extension
    )

    print("\n" + "=" * 65)
    print(f"FILE: {filepath}")
    print("=" * 65)

    if not os.path.exists(filepath):
        print("File not found!")
        return

    with open(filepath, "r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file):
            if line_number >= 10:
                break

            print(f"Line {line_number + 1}: {line.rstrip()}")


def main():

    for partition, trip_id in SAMPLE_TRIPS:

        for extension in EXTENSIONS:
            inspect_file(partition, trip_id, extension)


if __name__ == "__main__":
    main()
