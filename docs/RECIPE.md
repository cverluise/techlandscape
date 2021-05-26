# RECIPE

## Candidates

### Generate sample

````shell
dvc repro candidates
````

> :information_source: The annotation sample is randomly drawn from all candidates defined by the config file. Do not expect the exact same outs if you run it twice.

### Label sample

````shell
TECHNOLOGY="" # e.g. "additivemanufacturing"
prodigy textcat.option SEED_${TECHNOLOGY} data/candidates_${TECHNOLOGY}_sample.jsonl -F prodigy/textcat_option.py
````

### Load annotated SEED

```shell

# prep seed file
ls data/seed_*.jsonl | parallel 'mv {} {}.tmp && techlandscape utils add-accept-text {}.tmp >> {}'
rm data/seed_*.jsonl.tmp

# load to bq
cd data && ls seed_*.jsonl | parallel 'bq load --source_format NEWLINE_DELIMITED_JSON --replace --ignore_unknown_values --max_bad_records 1000 --autodetect patentcity:techdiffusion.{.} {} ../schemas/seed.json' && cd ../ 

```