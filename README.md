# Arctic and subarctic marine mammal call classification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22859088.svg)](https://doi.org/10.5281/zenodo.22859088)

A second one day machine learning demonstration built while applying for the
AWI/MarDATA PhD position in Data Science and Bioinformatics for Marine and
Arctic Biodiversity, this one aimed squarely at the posting's sequence data
and transformer based model preferences. Portfolio proof of concept, not a
submitted or peer reviewed study, see Limitations below. A companion project
on seabird diet composition covers the tabular, community ecology side of
the posting; this one covers sequence classification and a from scratch
transformer.

## What this shows

The posting lists sequence data analysis and transformer based models as
preferred skills, alongside required supervised and unsupervised machine
learning on biodiversity data. A short audio recording is a sequence in
exactly the sense a transformer is built for, whether the sequence is amino
acids, DNA bases, or, as here, spectrogram time frames, so this project uses
real marine mammal bioacoustic recordings as a concrete, honestly evaluated
stand in for the sequence based analysis (for example eDNA metabarcoding)
the position focuses on:

1. Loads and resamples 225 real WAV recordings across nine species to a
   common sample rate, the audio equivalent of a dataset harmonization step.
2. Extracts classical acoustic features (pooled MFCCs, deltas, and spectral
   shape descriptors) as a fixed length vector per clip.
3. Runs unsupervised clustering (PCA plus KMeans) to look for structure in
   that feature space.
4. Trains two random forest classifiers: one predicting species identity (9
   classes), one predicting broad taxonomic group (mysticete, odontocete, or
   pinniped, 3 classes) purely from the acoustic signal.
5. Trains a small transformer, from scratch, directly on log mel
   spectrograms for the taxonomic group task, and compares it honestly
   against the random forest on the identical held out split.

## Data

A 225 clip subset (25 per species) of the [Watkins Marine Mammal Sound
Database](https://cis.whoi.edu/science/B/whalesounds/index.cfm) (Woods Hole
Oceanographic Institution; Sayigh et al. 2016), sourced directly from the
public [Internet Archive
mirror](https://archive.org/details/watkins-marine-mammal-sound-database-full-cuts)
after the official site went down for maintenance while this project was
built. See `LICENSE_DATA.md` for full attribution. Six of the nine species
are genuinely Arctic or Subarctic (narwhal, beluga, bowhead whale, ringed
seal, bearded seal, walrus); killer whale and humpback whale are included as
wider ranging comparison species.

The audio itself is not committed to this repository. Only
`data_raw_manifest.csv`, a small text file listing which 225 clips make up
the subset, is included; running `analysis.py` downloads exactly those
clips directly from the archive mirror the first time it runs (a few
megabytes total), the same individual file extraction endpoint used to
build the subset in the first place, without ever pulling the full 6.2
gigabyte archive. This keeps the repository small and avoids redistributing
the recordings themselves under uncertain terms, since no formal license
statement was found on the mirror (see `LICENSE_DATA.md`).

## Results

- Unsupervised clustering on the classical acoustic feature space does not
  cleanly recover species identity or taxonomic group (adjusted Rand index
  0.04 for species, 0.00 for group). The PCA projection shows heavy overlap
  between groups in the two components with the most variance.
- Species classifier (9 way, random forest): 68 percent held out accuracy,
  75 percent 5 fold cross validated accuracy, from acoustic features alone.
- Taxonomic group classifier (3 way, random forest): 81 percent held out
  accuracy against a 33 percent majority baseline, 86 percent 5 fold cross
  validated accuracy. Spectral bandwidth and centroid, sensible acoustic
  correlates of the frequency differences between baleen whale moans,
  toothed whale clicks and whistles, and pinniped calls, are the top
  features.
- A small transformer trained from scratch on raw spectrograms, evaluated on
  the identical held out split as the group classifier above, reaches 84
  percent accuracy, in the same range as the random forest. This is a
  genuinely interesting single split result, not a robust benchmark: with
  only about 68 held out clips a few examples shift accuracy by several
  points, and unlike the random forest it is not cross validated. Read
  together with the unsupervised result, the pattern matches the seabird
  project: real signal that supervised methods, including a transformer,
  can recover even though it does not fall out of an unsupervised, low
  dimensional view of the same feature space.

Full numbers, confusion matrices, feature importances and all figures are in
the `outputs_*` and `figures_*` files, walked through with narrative in
`marine_mammal_acoustics_analysis.ipynb`. This repository was uploaded as
flat files through the browser for speed; running `analysis.py` locally
regenerates the same content into proper `outputs/` and `figures/` folders.

## Limitations

- 225 clips (25 per species) is small by deep learning standards, chosen
  deliberately to keep the project reproducible in a day without
  downloading the full 6.2 gigabyte archive. The transformer result is a
  proof of concept, not a robust benchmark.
- No pretrained audio or bioacoustic embedding model was used: general
  purpose audio transformer weights and existing bioacoustic model zoos
  (oriented mostly toward birds) were not reachable from this environment's
  network. The transformer here is trained entirely from scratch, the
  harder and more data hungry way to use one.
- Species and taxonomic group labels come directly from the database's own
  folder structure, not independently reverified against original recording
  metadata here.
- The three taxonomic groups are a simplification chosen to create a
  coarse, ecologically meaningful task, analogous to the era split in the
  seabird project, not a claim about the best way to group these species for
  a specific monitoring question.
- No environmental or contextual covariates (location, season, depth, ice
  cover) are used, only the audio signal itself.

## What real Arctic bioacoustic monitoring work would add

Fine tuning a pretrained audio or bioacoustic transformer instead of
training from scratch, scaling to the full Watkins database and to passive
acoustic monitoring archives from actual Arctic deployments, and applying
the same sequence classification framing used here for audio to eDNA
metabarcoding sequences, the sequence data type GBIF, OBIS and Arctic
biodiversity monitoring programmes increasingly rely on.

## Reproducing this

```
pip install -r requirements.txt
mkdir -p data_raw && mv data_raw_manifest.csv data_raw/manifest.csv
python analysis.py                     # downloads the 225 clips on first run, extracts features, runs all models, writes outputs/ and figures/
python build_notebook.py               # builds the notebook structure from analysis.py
jupyter nbconvert --to notebook --execute --inplace marine_mammal_acoustics_analysis.ipynb
```

## Repository layout

```
data_raw_manifest.csv   species, year and filename for the 225 clip subset (clips downloaded on first run, see Data above)
analysis.py              full pipeline: download, features, clustering, random forests, transformer
build_notebook.py        builds the notebook from the same functions in analysis.py
marine_mammal_acoustics_analysis.ipynb   narrated walkthrough with embedded figures and results
figures_*.png            all generated plots (analysis.py recreates these under figures/ when run locally)
outputs_*.csv / .txt     feature table, cluster assignments, classifier reports, model summary
LICENSE_DATA.md          data attribution and citation
```

## How to cite

> Frey, E. (2026). elifrey/marine-mammal-acoustics-ml: Arctic and subarctic marine mammal call classification (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.22859088

That DOI always resolves to the latest version. The v1.0.0 snapshot is 10.5281/zenodo.22859089.
