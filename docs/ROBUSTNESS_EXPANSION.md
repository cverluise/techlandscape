# ROBUSTNESS

## Expansion

### Question

How do random variations in the seed affect the overall expansion? If small variations in the seed (sample uncertainty) largely affect the overall expansion, this raises consistency issues.  

Related question: How to measure robustness of the expansion step?

### Approach 

We emulate data uncertainty by generating a sequence of random draws in the annotated seed. Then, we can compare the generated expansions and look at how data uncertainty affect the overall expansion. 

Using the generated expansions, we look at two things:

- pairwise overlap: for all *pairs* of expansions (k in n), we compute the share of families which are in *both* expansions and report moments of the distribution 
- batch overlap: for all *expansions*, we compute the share of families which are in *all* expansions and report moments of the distribution

### Results

Overall, we find that the expansion is robust to data uncertainty, even under particularly conservative conditions. Drawing 10 different random samples of the seed (50% of the seed), the resulting expansions exhibit a median family overlap ranging between 76% and 94%. The median (and other moments) of the overlap distribution grow as the share grows - as expected. 

??? quote "Code"
    
    ```shell
    dvc unprotect outs/expansion_*robustness*.csv
    dvc repro -f expansion-robustness
    dvc add outs/expansion_*robustness*.csv
    techlandscape robustness wrap-overlap-analysis "outs/expansion_*robustness*.csv" technologies --markdown 
    ```

#### additivemanufacturing

| configs               |   count |   mean |   std |   min |   25% |   50% |   75% |   max |
|:----------------------|--------:|-------:|------:|------:|------:|------:|------:|------:|
| batchrobustness0.5    |      10 |   0.76 |  0.08 |  0.62 |  0.71 |  0.76 |  0.82 |  0.86 |
| batchrobustness0.7    |      10 |   0.78 |  0.08 |  0.69 |  0.72 |  0.75 |  0.83 |  0.93 |
| batchrobustness0.9    |      10 |   0.79 |  0.06 |  0.71 |  0.78 |  0.79 |  0.79 |  0.94 |
| pairwiserobustness0.5 |      90 |   0.88 |  0.07 |  0.69 |  0.83 |  0.89 |  0.94 |  0.98 |
| pairwiserobustness0.7 |      90 |   0.91 |  0.07 |  0.72 |  0.88 |  0.93 |  0.97 |  0.99 |
| pairwiserobustness0.9 |      90 |   0.95 |  0.06 |  0.75 |  0.91 |  0.99 |  0.99 |  1    |

#### blockchain

| configs               |   count |   mean |   std |   min |   25% |   50% |   75% |   max |
|:----------------------|--------:|-------:|------:|------:|------:|------:|------:|------:|
| batchrobustness0.5    |      10 |   0.91 |  0.01 |  0.9  |  0.9  |  0.91 |  0.92 |  0.93 |
| batchrobustness0.7    |      10 |   0.94 |  0.01 |  0.92 |  0.93 |  0.94 |  0.94 |  0.95 |
| batchrobustness0.9    |      10 |   0.96 |  0    |  0.96 |  0.96 |  0.96 |  0.97 |  0.97 |
| pairwiserobustness0.5 |      90 |   0.96 |  0.01 |  0.93 |  0.95 |  0.96 |  0.97 |  0.98 |
| pairwiserobustness0.7 |      90 |   0.98 |  0.01 |  0.96 |  0.97 |  0.98 |  0.98 |  0.99 |
| pairwiserobustness0.9 |      90 |   0.99 |  0.01 |  0.97 |  0.98 |  0.99 |  0.99 |  1    |

#### computervision

| configs               |   count |   mean |   std |   min |   25% |   50% |   75% |   max |
|:----------------------|--------:|-------:|------:|------:|------:|------:|------:|------:|
| batchrobustness0.5    |      10 |   0.83 |  0.02 |  0.79 |  0.82 |  0.84 |  0.85 |  0.85 |
| batchrobustness0.7    |      10 |   0.85 |  0.01 |  0.84 |  0.85 |  0.85 |  0.86 |  0.87 |
| batchrobustness0.9    |      10 |   0.95 |  0    |  0.94 |  0.95 |  0.95 |  0.95 |  0.95 |
| pairwiserobustness0.5 |      90 |   0.92 |  0.02 |  0.87 |  0.91 |  0.92 |  0.94 |  0.96 |
| pairwiserobustness0.7 |      90 |   0.95 |  0.02 |  0.91 |  0.94 |  0.96 |  0.96 |  0.98 |
| pairwiserobustness0.9 |      90 |   0.99 |  0    |  0.98 |  0.99 |  0.99 |  0.99 |  1    |

#### genomeediting

| configs               |   count |   mean |   std |   min |   25% |   50% |   75% |   max |
|:----------------------|--------:|-------:|------:|------:|------:|------:|------:|------:|
| batchrobustness0.5    |      10 |   0.94 |  0.03 |  0.9  |  0.92 |  0.94 |  0.97 |  0.98 |
| batchrobustness0.7    |      10 |   0.97 |  0.01 |  0.94 |  0.97 |  0.97 |  0.98 |  0.98 |
| batchrobustness0.9    |      10 |   0.96 |  0.02 |  0.94 |  0.94 |  0.95 |  0.97 |  0.98 |
| pairwiserobustness0.5 |      90 |   0.97 |  0.02 |  0.92 |  0.95 |  0.98 |  0.99 |  0.99 |
| pairwiserobustness0.7 |      90 |   0.99 |  0.01 |  0.96 |  0.99 |  0.99 |  0.99 |  1    |
| pairwiserobustness0.9 |      90 |   0.99 |  0.02 |  0.95 |  0.99 |  0.99 |  1    |  1    |

#### hydrogenstorage

| configs               |   count |   mean |   std |   min |   25% |   50% |   75% |   max |
|:----------------------|--------:|-------:|------:|------:|------:|------:|------:|------:|
| batchrobustness0.5    |      10 |   0.9  |  0.01 |  0.88 |  0.89 |  0.9  |  0.91 |  0.91 |
| batchrobustness0.7    |      10 |   0.92 |  0.01 |  0.91 |  0.91 |  0.92 |  0.92 |  0.92 |
| batchrobustness0.9    |      10 |   0.96 |  0    |  0.95 |  0.96 |  0.96 |  0.96 |  0.97 |
| pairwiserobustness0.5 |      90 |   0.95 |  0.01 |  0.91 |  0.95 |  0.95 |  0.96 |  0.98 |
| pairwiserobustness0.7 |      90 |   0.97 |  0.01 |  0.95 |  0.97 |  0.97 |  0.98 |  0.99 |
| pairwiserobustness0.9 |      90 |   0.99 |  0.01 |  0.98 |  0.98 |  0.99 |  0.99 |  1    |

#### naturallanguageprocessing

| configs               |   count |   mean |   std |   min |   25% |   50% |   75% |   max |
|:----------------------|--------:|-------:|------:|------:|------:|------:|------:|------:|
| batchrobustness0.5    |      10 |   0.9  |  0.03 |  0.86 |  0.87 |  0.9  |  0.92 |  0.96 |
| batchrobustness0.7    |      10 |   0.92 |  0.01 |  0.91 |  0.92 |  0.92 |  0.92 |  0.96 |
| batchrobustness0.9    |      10 |   0.99 |  0    |  0.99 |  0.99 |  0.99 |  0.99 |  0.99 |
| pairwiserobustness0.5 |      90 |   0.96 |  0.03 |  0.89 |  0.94 |  0.96 |  0.98 |  0.99 |
| pairwiserobustness0.7 |      90 |   0.98 |  0.01 |  0.94 |  0.98 |  0.98 |  0.99 |  1    |
| pairwiserobustness0.9 |      90 |   1    |  0    |  0.99 |  0.99 |  1    |  1    |  1    |

#### selfdrivingvehicle

| configs               |   count |   mean |   std |   min |   25% |   50% |   75% |   max |
|:----------------------|--------:|-------:|------:|------:|------:|------:|------:|------:|
| batchrobustness0.5    |      10 |   0.9  |  0.02 |  0.87 |  0.88 |  0.89 |  0.91 |  0.92 |
| batchrobustness0.7    |      10 |   0.91 |  0.01 |  0.9  |  0.91 |  0.91 |  0.92 |  0.93 |
| batchrobustness0.9    |      10 |   0.93 |  0.01 |  0.92 |  0.93 |  0.93 |  0.94 |  0.94 |
| pairwiserobustness0.5 |      90 |   0.95 |  0.02 |  0.91 |  0.93 |  0.95 |  0.96 |  0.98 |
| pairwiserobustness0.7 |      90 |   0.97 |  0.01 |  0.93 |  0.96 |  0.97 |  0.97 |  0.98 |
| pairwiserobustness0.9 |      90 |   0.98 |  0.01 |  0.97 |  0.98 |  0.99 |  0.99 |  1    |