# Data attribution

The audio in `data_raw/` is 4,761 clips (nine species) of the Watkins Marine Mammal
Sound Database, built at Woods Hole Oceanographic Institution from decades
of field recordings originally collected by William A. Watkins, and later
digitized and organized into a public online database.

Standard citation for the database:

Sayigh, L., Daher, M. A., Allen, J., Gordon, H., Joyce, K., Stuhlmann, C.,
and Tyack, P. (2016). The Watkins Marine Mammal Sound Database: An online,
freely accessible resource. Proceedings of Meetings on Acoustics, 27,
040013.

The database's official home is
[cis.whoi.edu/science/B/whalesounds](https://cis.whoi.edu/science/B/whalesounds/index.cfm)
(also reachable at go.whoi.edu/marine-mammal-sounds), and it is described
there and in the citation above as a freely accessible resource for
research and education. That official site was down for scheduled
maintenance when this project was started, so the clips here were
fetched directly from the public Internet Archive mirror of the full
database,
[archive.org/details/watkins-marine-mammal-sound-database-full-cuts](https://archive.org/details/watkins-marine-mammal-sound-database-full-cuts),
selecting individual files from the server without downloading the full 6.2
gigabyte archive. No independent license statement was found on the mirror
itself; use here follows the terms described for the official database
(freely accessible for research and educational use), and anyone reusing
these clips beyond that should confirm current terms directly with Woods
Hole Oceanographic Institution.

`data_raw_manifest.csv` lists every clip's species, recording year, file
name and size as extracted from the archive. The audio itself is not
committed to this repository; `download_clips.py` downloads exactly these
4,761 clips directly from the mirror.

## My derived outputs

The source audio above is not mine and is reused under the terms of the Watkins
database. Everything I produced from it is.

The extracted acoustic feature tables, the derived result tables, the figures and
the trained model outputs in this repository are my own work and are released under
the Creative Commons Attribution 4.0 International Licence (CC BY 4.0):

https://creativecommons.org/licenses/by/4.0/

Reuse them freely, including commercially, with credit:

> Frey, E. (2026). Marine mammal call classification with random forest and transformer
> models on 4,761 Watkins recordings (v1.1.0). Zenodo. https://doi.org/10.5281/zenodo.22859088 ORCID 0009-0002-7482-7986

This CC BY 4.0 grant covers my derived outputs only. It does not extend to the raw
audio clips, which remain subject to the Watkins database terms above. The analysis
code is MIT licensed, see `LICENSE`.
