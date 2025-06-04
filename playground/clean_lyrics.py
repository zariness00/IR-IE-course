import pandas as pd
import re
import argparse 


def clean_lyrics(input_csv, output_csv):
    songs_file = pd.read_csv(input_csv)

    songs_file["Lyric"] = songs_file["Lyric"].apply(lambda lyrics_txt: re.sub(r"\[.*?\]", "", lyrics_txt))

    songs_file["Lyric"] = songs_file["Lyric"].apply(lambda lyrics_txt: "\n".join(line for line in lyrics_txt.splitlines() if line.strip()))

    songs_file.to_csv(output_csv, index=False)
    print(f"Cleaned lyrics saved to {output_csv}")

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

    clean_lyrics(args.input_csv, args.output_csv)


