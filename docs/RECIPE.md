# RECIPE

## Candidates

### Generate sample

````shell
ls configs/candidates_*.yaml | cut -d_ -f2 | parallel --eta 'techlandscape candidates get-candidates configs/candidates_{} techdiff.techdiff.candidates_{.} credentials_bq.json'
ls configs/candidates_*.yaml | cut -d_ -f2 | parallel --eta 'techlandscape candidates get-candidates-sample techdiff.techdiff.candidates_{.} credentials_bq.json --destination-table techdiff.techdiff.candidates_{.}_sample'
ls configs/candidates_*.yaml | cut -d_ -f2 | parallel --eta 'bq extract --destination_format NEWLINE_DELIMITED_JSON techdiff.techdiff.candidates_{.}_sample gs://tmp/candidates_{.}_sample.jsonl'
ls configs/candidates_*.yaml | cut -d/ -f2 | parallel --eta 'mv data/{.}_sample.jsonl data/{.}_sample.tmp.jsonl && techlandscape candidates prep-prodigy-annotation data/{.}_sample.tmp.jsonl configs/{} >> data/{.}_sample.jsonl' && rm data/*.tmp.*
````

### Label sample

````shell
TECHNOLOGY="" # e.g. "additivemanufacturing"
prodigy textcat.option SEED_${TECHNOLOGY} data/candidates_${TECHNOLOGY}_sample.jsonl -F prodigy/textcat_option.py
````

