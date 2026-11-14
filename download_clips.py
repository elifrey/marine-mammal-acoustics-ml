"""
Download the 4,761 Watkins clips listed in data_raw_manifest.csv, one file at
a time, from the public Internet Archive mirror of the Watkins Marine Mammal
Sound Database. Each clip is streamed out of the 6.2 GB archive individually,
so the full zip is never downloaded (about 710 MB in total).

    python download_clips.py            # writes data_raw/<species>/<year>/<file>.wav

Already downloaded clips are skipped, so the script can be rerun after an
interruption.
"""
import os
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

# archive.org redirects this to whichever storage server currently holds the item
ZIP_URL = ("https://archive.org/download/watkins-marine-mammal-sound-database-full-cuts/"
           "Watkins_Marine_Mammal_Sound_Database_full_cuts.zip/")
INNER = "Watkins_Marine_Mammal_Sound_Database"
OUT_DIR = "data_raw"
RETRIES = 3


def destination(row):
    return os.path.join(OUT_DIR, row.species, str(row.year), row.filename)


def download(row):
    """Fetch one clip into a .part file and rename it only when complete."""
    dest = destination(row)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = ZIP_URL + urllib.parse.quote(f"{INNER}/{row.species}/{row.year}/{row.filename}")
    for attempt in range(RETRIES):
        try:
            urllib.request.urlretrieve(url, dest + ".part")
            os.replace(dest + ".part", dest)
            return True
        except Exception as e:  # network errors: back off and retry
            if attempt == RETRIES - 1:
                print("failed:", dest, e, file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    return False


def main():
    manifest = pd.read_csv("data_raw_manifest.csv")
    todo = [row for _, row in manifest.iterrows() if not os.path.exists(destination(row))]
    print(f"{len(manifest)} clips in manifest, {len(todo)} to download")
    for i, row in enumerate(todo, 1):
        download(row)
        if i % 250 == 0:
            print(f"  {i}/{len(todo)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
