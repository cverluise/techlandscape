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
ls data/seed_*.jsonl | parallel 'mv {} {}.tmp && techlandscape utils add-practical-fields {}.tmp >> {}'
rm data/seed_*.jsonl.tmp

# load to bq
cd data && ls seed_*.jsonl | parallel 'bq load --source_format NEWLINE_DELIMITED_JSON --replace --ignore_unknown_values --max_bad_records 1000 --autodetect patentcity:techdiffusion.{.} {} ../schemas/seed.json' && cd ../
```

### Build publications table at family_id level

```shell
techlandscape assets get-publications-family patentcity.patents.publications credentials_bq.json

# fix REPEATED instead of NULLABLE nested field 
gsutil -m rm "gs://tmp/publications_*.jsonl.gz"
bq extract --destination_format NEWLINE_DELIMITED_JSON --compression GZIP patentcity:patents.publications "gs://tmp/publications_*.jsonl.gz" 
STAGE_FOLDER=""
gsutil -m cp "gs://tmp/publications_*.jsonl.gz" $STAGE_FOLDER
ls ${STAGE_FOLDER}/publications_*.jsonl.gz | parallel --eta 'mv {} {.}.tmp.gz && techlandscape utils flatten-nested-vars {.}.tmp.gz cpc,ipc,citation,cited_by >> {.} && gzip {.}'
gsutil -m cp "${STAGE_FOLDER}/publications_*.jsonl.gz" gs://tmp/
bq rm patentcity.patents.publications
bq load --source_format NEWLINE_DELIMITED_JSON --replace --max_bad_records 1000 patentcity:patents.publications "gs://tmp/publications_*.jsonl.gz" schemas/publications_familyid.json
```