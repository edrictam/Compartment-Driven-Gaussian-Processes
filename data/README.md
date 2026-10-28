# Frozen observation inputs

These four files were supplied with the recovered code and copied without changing their bytes.

- `warfarin.rda`, `theo_sd.rda`: R data objects attributed by the recovered notebooks to the `nlmixr2data` package. The recovered material does not specify the package version, source commit, or original download checksums. The supplied files are the frozen inputs used in these analyses.
- `5fu_cleaned.csv`, `capecitabine_cleaned.csv`: digitized cohort-mean observations attributed to Twelves et al. (1999), *Clinical Cancer Research* 5(7):1696–1702. Columns are `trajectory`, `x` (hours), `y` (concentration), and `se` (reported standard error, as labeled in the supplied code). `blank` identifies hepatic dysfunction and `filled` normal liver function according to the recovered notebook. The artificial zero-time rows have zero SE and are explicitly excluded by the analysis.

The original digitization project, source figure files, and digitization/export script were not supplied. Reproduction from these frozen observations is supported; independent reproduction of the digitization and confirmation of the error-bar interpretation require those source materials. The observation files are never replaced by live downloads during a run.
