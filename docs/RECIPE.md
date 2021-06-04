# RECIPE

## Candidates

### Generate sample

````shell
dvc repro candidates
````

!!! info
    The annotation sample is randomly drawn from all candidates defined by the config file. Do not expect the exact same outs if you run it twice.

### Label sample

````shell
TECHNOLOGY="" # e.g. "additivemanufacturing"
prodigy textcat.option SEED_${TECHNOLOGY} data/candidates_${TECHNOLOGY}_sample.jsonl -F prodigy/textcat_option.py
````

### Load annotated SEED

```shell
dvc unprotect data/seed*.jsonl
dvc repro load-annotated-seed
dvc add data/seed*.jsonl
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
bq rm patentcity:patents.publications
bq load --source_format NEWLINE_DELIMITED_JSON --replace --max_bad_records 1000 patentcity:patents.publications "gs://tmp/publications_*.jsonl.gz" schemas/publications_familyid.json
```

### Expansion iteration and robustness analysis

```shell
dvc unprotect outs/expansion_*robustness*.csv
dvc repro -f expansion-robustness
dvc add outs/expansion_*robustness*.csv
```

### Training data

```shell
cat lib/technology.txt | parallel --eta 'techlandscape io get-training-data family_id patentcity.techdiffusion.seed_{} patentcity.techdiffusion.expansion_{} 400 patentcity.techdiffusion.training_{} credentials_bq.json'
gsutil -m rm "gs://tmp/training_*.jsonl" 
cat lib/technology.txt | parallel --eta 'bq extract --destination_format NEWLINE_DELIMITED_JSON patentcity:techdiffusion.training_{} "gs://tmp/training_{}.jsonl" '
gsutil -m cp "gs://tmp/training_*.jsonl" data/
```