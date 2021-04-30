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

