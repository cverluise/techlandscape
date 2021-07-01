# MODEL ROBUSTNESS

## Question

How random variations in the seed and the antiseed affect the pruning step. 

## Approach

We emulate random variations by iterating over various training/test set randomly drawn from the same set of annotated data. We define a default model architecture, train as many models as training/test sets and compare the predictions of the different models out of sample (10,000) on a common benchmark of families/publications drawn from the expansion set.

Specifically, we look at:

1. moments of the distribution of standard errors of models predictions (std computed at the family/publication level)
1. models' consensus regarding positive and negative examples 

## Results

### additivemanufacturing - cnn

#### Score dispersion

|       |   std_score |
|:------|------------:|
| count |   10000     |
| mean  |       0.09  |
| std   |       0.054 |
| min   |       0     |
| 25%   |       0.055 |
| 50%   |       0.082 |
| 75%   |       0.113 |
| max   |       0.384 |

#### Models consensus

|   nb_models |   nb_positives |   share_positives |   cumshare_positives |
|------------:|---------------:|------------------:|---------------------:|
|           9 |            747 |             0.145 |                0.145 |
|           8 |            520 |             0.101 |                0.246 |
|           7 |            595 |             0.115 |                0.361 |
|           6 |            486 |             0.094 |                0.455 |
|           5 |            510 |             0.099 |                0.554 |
|           4 |            560 |             0.109 |                0.663 |
|           3 |            603 |             0.117 |                0.78  |
|           2 |            536 |             0.104 |                0.884 |
|           1 |            600 |             0.116 |                1     |
|           0 |              0 |             0     |                1     |

### additivemanufacturing - mlp

#### Score dispersion

|       |   std_score |
|:------|------------:|
| count |   10000     |
| mean  |       0.041 |
| std   |       0.039 |
| min   |       0     |
| 25%   |       0.014 |
| 50%   |       0.029 |
| 75%   |       0.056 |
| max   |       0.276 |

#### Models consensus

|   nb_models |   nb_positives |   share_positives |   cumshare_positives |
|------------:|---------------:|------------------:|---------------------:|
|          10 |            320 |             0.225 |                0.225 |
|           9 |            126 |             0.089 |                0.314 |
|           8 |            112 |             0.079 |                0.393 |
|           7 |             42 |             0.03  |                0.423 |
|           6 |            114 |             0.08  |                0.503 |
|           5 |             95 |             0.067 |                0.57  |
|           4 |            124 |             0.087 |                0.657 |
|           3 |            129 |             0.091 |                0.748 |
|           2 |            152 |             0.107 |                0.855 |
|           1 |            206 |             0.145 |                1     |
|           0 |              0 |             0     |                1     |

### blockchain - cnn

#### Score dispersion

|       |   std_score |
|:------|------------:|
| count |   10000     |
| mean  |       0.074 |
| std   |       0.073 |
| min   |       0     |
| 25%   |       0.012 |
| 50%   |       0.047 |
| 75%   |       0.126 |
| max   |       0.389 |

#### Models consensus

|   nb_models |   nb_positives |   share_positives |   cumshare_positives |
|------------:|---------------:|------------------:|---------------------:|
|          10 |            410 |             0.084 |                0.084 |
|           9 |            360 |             0.074 |                0.158 |
|           8 |            352 |             0.072 |                0.23  |
|           7 |            336 |             0.069 |                0.299 |
|           6 |            360 |             0.074 |                0.373 |
|           5 |            390 |             0.08  |                0.453 |
|           4 |            380 |             0.078 |                0.531 |
|           3 |            357 |             0.073 |                0.604 |
|           2 |           1410 |             0.289 |                0.893 |
|           1 |            522 |             0.107 |                1     |
|           0 |              0 |             0     |                1     |

### blockchain - mlp

#### Score dispersion

|       |   std_score |
|:------|------------:|
| count |   10000     |
| mean  |       0.025 |
| std   |       0.04  |
| min   |       0     |
| 25%   |       0.003 |
| 50%   |       0.008 |
| 75%   |       0.03  |
| max   |       0.272 |

#### Models consensus

|   nb_models |   nb_positives |   share_positives |   cumshare_positives |
|------------:|---------------:|------------------:|---------------------:|
|          10 |            160 |             0.179 |                0.179 |
|           9 |             18 |             0.02  |                0.199 |
|           8 |             24 |             0.027 |                0.226 |
|           7 |             42 |             0.047 |                0.273 |
|           6 |             66 |             0.074 |                0.346 |
|           5 |            125 |             0.14  |                0.486 |
|           4 |             56 |             0.063 |                0.549 |
|           3 |            105 |             0.117 |                0.666 |
|           2 |            122 |             0.136 |                0.802 |
|           1 |            177 |             0.198 |                1     |
|           0 |              0 |             0     |                1     |

### computervision - cnn

#### Score dispersion

|       |   std_score |
|:------|------------:|
| count |   10000     |
| mean  |       0.065 |
| std   |       0.074 |
| min   |       0     |
| 25%   |       0.003 |
| 50%   |       0.029 |
| 75%   |       0.121 |
| max   |       0.383 |

#### Models consensus

|   nb_models |   nb_positives |   share_positives |   cumshare_positives |
|------------:|---------------:|------------------:|---------------------:|
|          10 |           3590 |             0.381 |                0.381 |
|           9 |            810 |             0.086 |                0.467 |
|           8 |            504 |             0.054 |                0.521 |
|           7 |            483 |             0.051 |                0.572 |
|           6 |            468 |             0.05  |                0.622 |
|           5 |            400 |             0.042 |                0.664 |
|           4 |            340 |             0.036 |                0.7   |
|           3 |            330 |             0.035 |                0.735 |
|           2 |           2072 |             0.22  |                0.955 |
|           1 |            420 |             0.045 |                1     |
|           0 |              0 |             0     |                1     |

### computervision - mlp

#### Score dispersion

|       |   std_score |
|:------|------------:|
| count |   10000     |
| mean  |       0.035 |
| std   |       0.046 |
| min   |       0     |
| 25%   |       0.006 |
| 50%   |       0.015 |
| 75%   |       0.042 |
| max   |       0.291 |

#### Models consensus

|   nb_models |   nb_positives |   share_positives |   cumshare_positives |
|------------:|---------------:|------------------:|---------------------:|
|          10 |           3310 |             0.55  |                0.55  |
|           9 |            684 |             0.114 |                0.664 |
|           8 |            360 |             0.06  |                0.724 |
|           7 |            357 |             0.059 |                0.783 |
|           6 |            300 |             0.05  |                0.833 |
|           5 |            225 |             0.037 |                0.87  |
|           4 |            212 |             0.035 |                0.906 |
|           3 |            204 |             0.034 |                0.939 |
|           2 |            180 |             0.03  |                0.969 |
|           1 |            184 |             0.031 |                1     |
|           0 |              0 |             0     |                1     |

### genomeediting - cnn

#### Score dispersion

|       |   std_score |
|:------|------------:|
| count |   10000     |
| mean  |       0.03  |
| std   |       0.053 |
| min   |       0     |
| 25%   |       0     |
| 50%   |       0.001 |
| 75%   |       0.023 |
| max   |       0.329 |

#### Models consensus

|   nb_models |   nb_positives |   share_positives |   cumshare_positives |
|------------:|---------------:|------------------:|---------------------:|
|          10 |            450 |             0.126 |                0.126 |
|           9 |             90 |             0.025 |                0.152 |
|           8 |             88 |             0.025 |                0.176 |
|           7 |             70 |             0.02  |                0.196 |
|           6 |             96 |             0.027 |                0.223 |
|           5 |             90 |             0.025 |                0.248 |
|           4 |            280 |             0.079 |                0.327 |
|           3 |           1866 |             0.524 |                0.851 |
|           2 |            356 |             0.1   |                0.951 |
|           1 |            175 |             0.049 |                1     |
|           0 |              0 |             0     |                1     |

### genomeediting - mlp

#### Score dispersion

|       |   std_score |
|:------|------------:|
| count |   10000     |
| mean  |       0.011 |
| std   |       0.021 |
| min   |       0     |
| 25%   |       0.001 |
| 50%   |       0.003 |
| 75%   |       0.01  |
| max   |       0.205 |

#### Models consensus

|   nb_models |   nb_positives |   share_positives |   cumshare_positives |
|------------:|---------------:|------------------:|---------------------:|
|          10 |            360 |             0.462 |                0.462 |
|           9 |             63 |             0.081 |                0.543 |
|           8 |             56 |             0.072 |                0.615 |
|           7 |             28 |             0.036 |                0.651 |
|           6 |             60 |             0.077 |                0.728 |
|           5 |             40 |             0.051 |                0.779 |
|           4 |             60 |             0.077 |                0.856 |
|           3 |             33 |             0.042 |                0.899 |
|           2 |             48 |             0.062 |                0.96  |
|           1 |             31 |             0.04  |                1     |
|           0 |              0 |             0     |                1     |

### hydrogenstorage - cnn

#### Score dispersion

|       |   std_score |
|:------|------------:|
| count |   10000     |
| mean  |       0.046 |
| std   |       0.04  |
| min   |       0     |
| 25%   |       0.011 |
| 50%   |       0.037 |
| 75%   |       0.076 |
| max   |       0.311 |

#### Models consensus

|   nb_models |   nb_positives |   share_positives |   cumshare_positives |
|------------:|---------------:|------------------:|---------------------:|
|          10 |            230 |             0.213 |                0.213 |
|           9 |            117 |             0.108 |                0.321 |
|           8 |            120 |             0.111 |                0.432 |
|           7 |             91 |             0.084 |                0.516 |
|           6 |             90 |             0.083 |                0.599 |
|           5 |             75 |             0.069 |                0.669 |
|           4 |            100 |             0.093 |                0.761 |
|           3 |             72 |             0.067 |                0.828 |
|           2 |             84 |             0.078 |                0.906 |
|           1 |            102 |             0.094 |                1     |
|           0 |              0 |             0     |                1     |

### hydrogenstorage - mlp

#### Score dispersion

|       |   std_score |
|:------|------------:|
| count |   10000     |
| mean  |       0.026 |
| std   |       0.031 |
| min   |       0.001 |
| 25%   |       0.007 |
| 50%   |       0.015 |
| 75%   |       0.03  |
| max   |       0.257 |

#### Models consensus

|   nb_models |   nb_positives |   share_positives |   cumshare_positives |
|------------:|---------------:|------------------:|---------------------:|
|          10 |            160 |             0.21  |                0.21  |
|           9 |             54 |             0.071 |                0.281 |
|           8 |             40 |             0.052 |                0.333 |
|           7 |             63 |             0.083 |                0.416 |
|           6 |             78 |             0.102 |                0.518 |
|           5 |             65 |             0.085 |                0.604 |
|           4 |             80 |             0.105 |                0.709 |
|           3 |             60 |             0.079 |                0.787 |
|           2 |             72 |             0.094 |                0.882 |
|           1 |             90 |             0.118 |                1     |
|           0 |              0 |             0     |                1     |

### naturallanguageprocessing - cnn

#### Score dispersion

|       |   std_score |
|:------|------------:|
| count |   10000     |
| mean  |       0.058 |
| std   |       0.072 |
| min   |       0     |
| 25%   |       0.001 |
| 50%   |       0.018 |
| 75%   |       0.114 |
| max   |       0.39  |

#### Models consensus

|   nb_models |   nb_positives |   share_positives |   cumshare_positives |
|------------:|---------------:|------------------:|---------------------:|
|          10 |           1630 |             0.199 |                0.199 |
|           9 |            666 |             0.081 |                0.281 |
|           8 |            520 |             0.064 |                0.344 |
|           7 |            301 |             0.037 |                0.381 |
|           6 |            354 |             0.043 |                0.424 |
|           5 |            475 |             0.058 |                0.482 |
|           4 |           2972 |             0.363 |                0.846 |
|           3 |            525 |             0.064 |                0.91  |
|           2 |            384 |             0.047 |                0.957 |
|           1 |            353 |             0.043 |                1     |
|           0 |              0 |             0     |                1     |

### naturallanguageprocessing - mlp

#### Score dispersion

|       |   std_score |
|:------|------------:|
| count |   10000     |
| mean  |       0.028 |
| std   |       0.036 |
| min   |       0     |
| 25%   |       0.004 |
| 50%   |       0.012 |
| 75%   |       0.036 |
| max   |       0.339 |

#### Models consensus

|   nb_models |   nb_positives |   share_positives |   cumshare_positives |
|------------:|---------------:|------------------:|---------------------:|
|          10 |            950 |             0.368 |                0.368 |
|           9 |            315 |             0.122 |                0.49  |
|           8 |            216 |             0.084 |                0.574 |
|           7 |            231 |             0.09  |                0.663 |
|           6 |            204 |             0.079 |                0.742 |
|           5 |            170 |             0.066 |                0.808 |
|           4 |            120 |             0.046 |                0.855 |
|           3 |             96 |             0.037 |                0.892 |
|           2 |            126 |             0.049 |                0.941 |
|           1 |            153 |             0.059 |                1     |
|           0 |              0 |             0     |                1     |

### selfdrivingvehicle - cnn

#### Score dispersion

|       |   std_score |
|:------|------------:|
| count |   10000     |
| mean  |       0.085 |
| std   |       0.044 |
| min   |       0     |
| 25%   |       0.052 |
| 50%   |       0.091 |
| 75%   |       0.111 |
| max   |       0.324 |

#### Models consensus

|   nb_models |   nb_positives |   share_positives |   cumshare_positives |
|------------:|---------------:|------------------:|---------------------:|
|          10 |            610 |             0.086 |                0.086 |
|           9 |            765 |             0.107 |                0.193 |
|           8 |            504 |             0.071 |                0.263 |
|           7 |            483 |             0.068 |                0.331 |
|           6 |            468 |             0.066 |                0.397 |
|           5 |            390 |             0.055 |                0.451 |
|           4 |            488 |             0.068 |                0.52  |
|           3 |            423 |             0.059 |                0.579 |
|           2 |           2292 |             0.321 |                0.9   |
|           1 |            711 |             0.1   |                1     |
|           0 |              0 |             0     |                1     |

### selfdrivingvehicle - mlp

#### Score dispersion

|       |   std_score |
|:------|------------:|
| count |   10000     |
| mean  |       0.049 |
| std   |       0.036 |
| min   |       0.002 |
| 25%   |       0.021 |
| 50%   |       0.039 |
| 75%   |       0.069 |
| max   |       0.278 |

#### Models consensus

|   nb_models |   nb_positives |   share_positives |   cumshare_positives |
|------------:|---------------:|------------------:|---------------------:|
|          10 |           1210 |             0.302 |                0.302 |
|           9 |            414 |             0.104 |                0.406 |
|           8 |            336 |             0.084 |                0.49  |
|           7 |            294 |             0.074 |                0.564 |
|           6 |            276 |             0.069 |                0.633 |
|           5 |            335 |             0.084 |                0.716 |
|           4 |            308 |             0.077 |                0.793 |
|           3 |            243 |             0.061 |                0.854 |
|           2 |            266 |             0.066 |                0.92  |
|           1 |            318 |             0.08  |                1     |
|           0 |              0 |             0     |                1     |