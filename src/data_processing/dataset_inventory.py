
import os
import csv

DATASET_DIR = "data"

REQUIRED_EXTENSIONS = [
    ".track",
    ".nodes",
    ".arcs",
    ".route"
]


def inspect_partition(partition_path, partition_name):

    records = []

    if not os.path.exists(partition_path):
        print(f"Missing directory: {partition_path}")
        return records

    for trip_id in sorted(os.listdir(partition_path)):

        trip_path = os.path.join(partition_path, trip_id)

        if not os.path.isdir(trip_path):
            continue

        record = {
            "partition": partition_name,
            "trip_id": trip_id
        }

        for extension in REQUIRED_EXTENSIONS:

            filename = trip_id + extension
            filepath = os.path.join(trip_path, filename)

            exists = os.path.isfile(filepath)

            record[extension[1:] + "_exists"] = exists

            if exists:
                record[extension[1:] + "_size_kb"] = round(
                    os.path.getsize(filepath) / 1024, 2
                )
            else:
                record[extension[1:] + "_size_kb"] = 0

        record["complete"] = all(
            record[extension[1:] + "_exists"]
            for extension in REQUIRED_EXTENSIONS
        )

        records.append(record)

    return records


def main():

    all_records = []

    all_records.extend(
        inspect_partition(
            os.path.join(DATASET_DIR, "train"),
            "train"
        )
    )

    all_records.extend(
        inspect_partition(
            os.path.join(DATASET_DIR, "test"),
            "test"
        )
    )

    if not all_records:
        print("No trip folders found. Check dataset location.")
        return

    os.makedirs("results/eda", exist_ok=True)

    output_path = "results/eda/dataset_inventory.csv"

    with open(output_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=all_records[0].keys()
        )

        writer.writeheader()
        writer.writerows(all_records)

    print("\nDATASET INVENTORY COMPLETED")
    print("---------------------------")
    print("Total trips:", len(all_records))

    for partition in ["train", "test"]:

        subset = [
            r for r in all_records
            if r["partition"] == partition
        ]

        complete = sum(r["complete"] for r in subset)

        print(f"{partition} trips: {len(subset)}")
        print(f"Complete {partition} trips: {complete}")

    print("\nOutput saved to:", output_path)


if __name__ == "__main__":
    main()
