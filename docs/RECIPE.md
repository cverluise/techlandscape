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

### Prediction robustness

```shell
parallel --eta 'techlandscape utils train-test-split data/training_{1}.jsonl data/train_{1}_{2}.jsonl data/test_{1}_{2}.jsonl' ::: $(cat lib/technology.txt) ::: $(pwgen 5 10)
parallel --eta 'python techlandscape/model.py +model_{3}=default data.train=data/train_{1}_{2}.jsonl data.test=data/test_{1}_{2}.jsonl out=models/{1}_{2}_{3} logger.tensorboard.logdir=models/{1}_{2}_{3}/log' ::: $(cat lib/technology.txt) ::: $(ls data/test_*.jsonl | cut -d_ -f3 | cut -d. -f1 | sort -u) ::: cnn mlp
cat lib/technology.txt | parallel --eta 'techlandscape io get-expansion family_id patentcity.techdiffusion.expansion_{} patentcity.stage.expansion_{}_sample credentials_bq.json --sample-size 10000'
gsutil -m rm "gs://tmp/expansion_*_sample.jsonl.gz"
cat lib/technology.txt | parallel --eta 'bq extract --destination_format NEWLINE_DELIMITED_JSON --compression GZIP patentcity:stage.expansion_{}_sample "gs://tmp/expansion_{}_sample.jsonl.gz"'
gsutil -m cp "gs://tmp/expansion_*_sample.jsonl.gz" data/ 

parallel --eta 'techlandscape robustness get-prediction-analysis "models/{1}_*_{2}/model-best" data/expansion_{1}_sample.jsonl --destination outs/' ::: $(cat lib/technology.txt) ::: $(cat lib/model.txt)
parallel -j1 'techlandscape robustness wrap-prediction-analysis --group-by-technology "outs/classification_{1}_*.csv" >> docs/ROBUSTNESS_MODEL.md' ::: $(cat lib/technology.txt )
parallel --eta -j1 'techlandscape evaluate models-performance "models/{1}_*_{2}/model-best/meta.json" --markdown --title "{1} - {2}"' ::: $(cat lib/technology.txt) ::: $(cat lib/model.txt) >> docs/MODELS_PERFORMANCE.md 
```

### Transformers

```shell
gsutil -m rm -rf gs://techlandscape_tmp
gsutil mb gs://techlandscape_tmp
# prep data and models. Nb: could be imported via dvc but takes longer
ls data | grep -v dvc | grep 'train_\|test_\|expansion_' | parallel -j10 --eta 'gsutil cp data/{} gs://techlandscape_tmp/data/'
gsutil -m cp models/ gs://techlandscape_tmp/models/
# Then run notebook

# Get back performance enriched metrics
for TECH in $(cat lib/technology.txt); do
  for SAMPLE in $(cat lib/sample.txt); do
    gsutil cp "gs://techlandscape_tmp/models/${TECH}_${SAMPLE}_transformers/model-last/meta.json" models/${TECH}_${SAMPLE}_transformers/model-last/meta.json
  done;
done;
```

### Pruning

```shell
# Extract expansion table
parallel --eta 'techlandscape io join patentcity.techdiffusion.expansion_{1} patentcity.patents.family_abstract family_id patentcity.stage.expansion_{1} credentials_bq.json --abstract2text' ::: $(cat lib/technology.txt)
gsutil -m rm "gs://tmp/expansion_*.jsonl.gz"
parallel --eta 'bq extract --destination_format NEWLINE_DELIMITED_JSON --compression GZIP patentcity:stage.expansion_{1} "gs://tmp/expansion_{1}_*.jsonl.gz"' ::: $(cat lib/technology.txt)
gsutil -m cp "gs://tmp/expansion_*.jsonl.gz" data_tmp/
# check for encoding errors (opt)
#techlandscape utils inpsect-jsonl "data_tmp/expansion_*.jsonl.gz"

for MODEL in $(cat lib/models_selection.txt | grep -v transformers); 
  do DEST=$(echo "$MODEL" | cut -d/ -f2) &&
  TECH=$(echo "$DEST" | cut -d_ -f1) && 
  echo ${DEST} && echo ${TECH} &&
  techlandscape pruning get-score "data_tmp/expansion_${TECH}_*.jsonl.gz" ${MODEL} "data/${DEST}.jsonl"; 
done;
```