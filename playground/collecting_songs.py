from lyricsgenius import Genius
import csv
import re

# Paste your Genius API key here
genius = Genius("HSO1QJNfOE2xFbWoQCaoBmjZchaSrg5WdPp9i9XjVJxw9y7RM9S5LQE8iTWUPAq2")
api_key = "HSO1QJNfOE2xFbWoQCaoBmjZchaSrg5WdPp9i9XjVJxw9y7RM9S5LQE8iTWUPAq2"
# song = genius.search_song("Busy Woman", "Sabrina Carpenter", get_full_info=True)
# song_dict = song.to_dict()
# print("!")

# print(song_dict)
# print("!")
# print(song_dict["release_date_components"]["year"])
# print(song_dict['album']["name"])
# print(song_dict["lyrics_state"])
# print("!!!!!!")
# raw_lyrics = song_dict["lyrics"]
# intro_match = re.search(r'\[Intro\]', raw_lyrics, re.IGNORECASE)
# if intro_match:
#     # Slice from the start of "[Intro]"
#     trimmed_lyrics = raw_lyrics[intro_match.start():].strip()
#     print(trimmed_lyrics)
# else:
#     trimmed_lyrics = raw_lyrics.strip()

# print("!!!!!!")
def fetch_and_save_songs(artist_name, max_songs, output_csv):
    """
    Uses lyricsgenius to fetch up to `max_songs` by `artist_name` and saves
    artist, title, album, release date, release year, and lyrics to output_csv.
    """
    genius = Genius(api_key, timeout=15, remove_section_headers=False)
    
    # Search for the artist and get up to max_songs
    artist = genius.search_artist(artist_name, max_songs=max_songs, sort="popularity", include_features=False)
    if artist is None:
        print(f"No data found for artist '{artist_name}'.")
        return

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Artist", "Title", "Album", "Lyric", "Year"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for song in artist.songs:
            # song.artist → artist name (string)
            # song.title → title (string)
            # song.album → album name (string) or None
            # song.release_date → full date string (e.g., "August 29, 2024") or ""
            # song.lyrics → full lyrics (string)
            # Convert the song object to a dict to access release_date_components
            song_dict = song.to_dict()

            # Extract release year from release_date_components if available
            # year = ""
            # rd_components = song_dict.get("release_date_components", {})
            # if rd_components and rd_components.get("year"):
            #     year = str(rd_components["year"])

            #year = song_dict["relase_date_components"]["year"] 

            rd_components = song_dict.get("release_date_components") or {}
            year = str(rd_components.get("year", ""))
            album_name = "Unknown Album"
            album_info = song_dict.get("album")
            if album_info and album_info.get("name"):
                album_name = album_info["name"]
            
            # We don't need a full date string, so leave 'date' empty
            raw_lyrics = song_dict.get("lyrics")

            # Use regex to remove everything before "[Intro]" (case-insensitive)
            # If "[Intro]" is not present, keep lyrics as-is
            match = re.search(r'\[(Intro|Verse\s*1)\]', raw_lyrics, re.IGNORECASE)
            #intro_match = re.search(r'(?i)^.*?"?read\s*more', raw_lyrics)
            if match:
                # Slice from the start of "[Intro]"
                trimmed_lyrics = raw_lyrics[match.start():].strip()
            else:
                trimmed_lyrics = raw_lyrics.strip()
            writer.writerow({
                "Artist": song_dict.get("artist", artist_name),
                "Title": song_dict.get("title", ""),
                "Album": album_name,
                "Lyric": trimmed_lyrics,
                "Year": year,
                
            })
           
    print(f"Saved {len(artist.songs)} songs to '{output_csv}'.")


if __name__ == "__main__":
    # Example usage:
    artist_name = "Sabrina Carpenter"
    max_songs = 100
    output_csv = "sabrina_carpenter_lyrics.csv"
    
    fetch_and_save_songs(artist_name, max_songs, output_csv)
