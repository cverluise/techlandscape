# RECIPE

## Candidates

````shell
BQ_DATASET="" # e.g. techdiff.techdiff
GS_STAGE="" # e.g. gs://tmp
ls configs/candidates_*.yaml | cut -d_ -f2 | parallel --eta 'techlandscape candidates get-candidates configs/candidates_{} techdiff.techdiff.candidates_{.} credentials_bq.json'
ls configs/candidates_*.yaml | cut -d_ -f2 | parallel --eta 'techlandscape candidates get-candidates-sample techdiff.techdiff.candidates_{.} credentials_bq.json --destination-table techdiff.techdiff.candidates_{.}_sample'
ls configs/candidates_*.yaml | cut -d_ -f2 | parallel --eta 'bq extract --destination_format NEWLINE_DELIMITED_JSON techdiff.techdiff.candidates_{.}_sample gs://tmp/candidates_{.}_sample.jsonl'
````

