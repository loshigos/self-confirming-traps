Place the Duolingo learner sample here as `duolingo_sample.csv.gz`.

The raw data are the Duolingo spaced-repetition logs released with half-life regression
(Settles & Meeder, 2016), available from Harvard Dataverse: https://doi.org/10.7910/DVN/N8XJME
They are not redistributed in this repository.

To draw the sample used in the paper, run `code/duolingo_muestra_colab.py` (Google Colab or any
Python environment). It downloads the file, keeps all records of a hash-selected subset of learners
(seed "caro2027") and writes `duolingo_muestra_1M.csv.gz`; rename it to `duolingo_sample.csv.gz`.
