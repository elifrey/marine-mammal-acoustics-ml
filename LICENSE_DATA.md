# Data attribution

The audio in `data_raw/` is a 225 clip subset of the Watkins Marine Mammal
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
maintenance while this project was built, so the 225 clips here were
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
committed to this repository; `analysis.py` downloads exactly these 225
clips directly from the mirror the first time it runs.
