"""
Download the 4,761 Watkins clips listed in data_raw_manifest.csv, one file at
a time, from the public Internet Archive mirror of the Watkins Marine Mammal
Sound Database. Each clip is streamed out of the 6.2 GB archive individually,
so the full zip is never downloaded (about 710 MB in total).

    python download_clips.py            # writes data_raw/<species>/<year>/<file>.wav

Already downloaded clips are skipped, so the script can be rerun after an
interruption.
"""
import os, sys, time, urllib.parse, urllib.request
import pandas as pd

# archive.org redirects this to whichever storage server currently holds the item
ZIP_URL = ("https://archive.org/download/watkins-marine-mammal-sound-database-full-cuts/"
           "Watkins_Marine_Mammal_Sound_Database_full_cuts.zip/")
INNER = "Watkins_Marine_Mammal_Sound_Database"
OUT = "data_raw"

manifest = pd.read_csv("data_raw_manifest.csv")
todo = [r for _, r in manifest.iterrows()
        if not os.path.exists(os.path.join(OUT, r.species, str(r.year), r.filename))]
print(f"{len(manifest)} clips in manifest, {len(todo)} to download")
for i, r in enumerate(todo, 1):
    dest = os.path.join(OUT, r.species, str(r.year), r.filename)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = ZIP_URL + urllib.parse.quote(f"{INNER}/{r.species}/{r.year}/{r.filename}")
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, dest + ".part"); os.replace(dest + ".part", dest); break
        except Exception as e:
            if attempt == 2: print("failed:", dest, e, file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    if i % 250 == 0: print(f"  {i}/{len(todo)}", flush=True)
print("done")
