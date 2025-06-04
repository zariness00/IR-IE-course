import pandas as pd
import argparse

def drop_columns(input_csv, output_csv):
    songs_file = pd.read_csv(input_csv)

    songs_file = songs_file.drop(columns = ["Unnamed: 0", "Date"], axis=1) 
    songs_file.to_csv(output_csv, index=False)
    print(f"Cleaned dataset saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_csv",
        help="Path to the input CSV"
    )
    parser.add_argument(
        "output_csv",
        help="Path where the cleaned CSV should be written"
    )
    args = parser.parse_args()

    drop_columns(args.input_csv, args.output_csv)


