# Results 

### Comparison between all strategy baseline and ConFAR
![Strategies_FER_forgetting](https://github.com/user-attachments/assets/ab110219-99b3-4f03-9192-0949a86df651)

### Bar chart comparison for baselines with and without causality
![Bar_Chart_AU](https://github.com/user-attachments/assets/2a106b52-d599-4215-bc90-593a3b680850)

### Line plots for different sstrategies with and without Causality
![UB](https://github.com/user-attachments/assets/6e7f905a-0389-4dfe-8797-6a85f23b571b)
![SI](https://github.com/user-attachments/assets/b937796d-dfa4-4676-b5a0-8e5dc3b2d1f4)
![NR](https://github.com/user-attachments/assets/19abf856-e8ce-48d8-b7ee-3293b63a03da)
![MAS](https://github.com/user-attachments/assets/e9dce817-5d8b-429e-82ae-38c9e8389681)
![LB](https://github.com/user-attachments/assets/360a12e3-a99a-4adc-8ee1-a9b8819f625a)
![EWC](https://github.com/user-attachments/assets/8bc82d91-cfaa-4424-857a-b09dbb858142)


### Line plots for different sstrategies with and without Causality (together for FER and AU)
![SI](https://github.com/user-attachments/assets/e876efaa-adc9-4319-b682-54eefe476b5e)
![NR](https://github.com/user-attachments/assets/5e44173c-8ca6-417e-b3e8-6b52e29022dc)
![MAS](https://github.com/user-attachments/assets/6e00035c-1c80-4337-aee9-ecd35a6d42af)
![LB](https://github.com/user-attachments/assets/aca05795-0035-419c-9022-1f234689b328)
![EWC](https://github.com/user-attachments/assets/be8e8b8b-a2ec-4546-88c0-fdbcc3ebc977)
![UB](https://github.com/user-attachments/assets/aea6e6ad-086c-4610-9594-5788444d35c6)

# Here's the result on which these plots are based.
(The *Task accuracy-dictionairy* is in the results.xslx file)

## Results Without Causality
### a) For task order: AU AV FER

#### LB
| Train/Test | AU       | AV       | FER        |
|------------|----------|----------|------------|
| AU         | 0.805495 | N/A      | N/A        |
| AV         | 0.810136 | 0.874595 | N/A        |
| FER        | 0.685172 | 0.075042 | 97.010313  |

#### EWC
| Train/Test | AU       | AV       | FER        |
|------------|----------|----------|------------|
| AU         | 0.855726 | N/A      | N/A        |
| AV         | 0.806158 | 0.854308 | N/A        |
| FER        | 0.628779 | 0.087046 | 96.236612  |

#### MAS
| Train/Test | AU       | AV       | FER        |
|------------|----------|----------|------------|
| AU         | 0.815495 | N/A      | N/A        |
| AV         | 0.806812 | 0.895075 | N/A        |
| FER        | 0.691359 | 0.105186 | 95.912552  |

#### SI
| Train/Test | AU       | AV       | FER        |
|------------|----------|----------|------------|
| AU         | 0.808686 | N/A      | N/A        |
| AV         | 0.805728 | 0.877147 | N/A        |
| FER        | 0.701757 | 0.109420 | 96.966228  |

#### NR
| Train/Test | AU       | AV       | FER        |
|------------|----------|----------|------------|
| AU         | 0.808059 | N/A      | N/A        |
| AV         | 0.863524 | 0.872325 | N/A        |
| FER        | 0.824291 | 0.618884 | 77.692333  |

#### UB
| Train/Test | AU       | AV       | FER        |
|------------|----------|----------|------------|
| AU         | 0.843321 | N/A      | N/A        |
| AV         | 0.860321 | 0.837831 | N/A        |
| FER        | 0.806023 | 0.683205 | 78.174007  |

### b) For task order: FER AV AU

#### LB
| Train/Test | FER       | AV       | AU         |
|------------|-----------|----------|------------|
| FER        | 97.797479 | N/A      | N/A        |
| AV         | 38.176594 | 0.914197 | N/A        |
| AU         | 2.218848  | 0.022769 | 0.805495   |

#### EWC
| Train/Test | FER       | AV       | AU         |
|------------|-----------|----------|------------|
| FER        | 97.475836 | N/A      | N/A        |
| AV         | 27.416405 | 0.902924 | N/A        |
| AU         | 3.575627  | 0.036849 | 0.826011   |

#### MAS
| Train/Test | FER       | AV       | AU         |
|------------|-----------|----------|------------|
| FER        | 97.642372 | N/A      | N/A        |
| AV         | 46.314982 | 0.903494 | N/A        |
| AU         | 16.898838 | 0.066291 | 0.857536   |

#### SI
| Train/Test | FER       | AV       | AU         |
|------------|-----------|----------|------------|
| FER        | 97.534613 | N/A      | N/A        |
| AV         | 36.386494 | 0.904588 | N/A        |
| AU         | 26.216366 | 0.197619 | 0.861999   |

#### NR
| Train/Test | FER       | AV       | AU         |
|------------|-----------|----------|------------|
| FER        | 97.732631 | N/A      | N/A        |
| AV         | 52.136844 | 0.903197 | N/A        |
| AU         | 19.842833 | 0.325731 | 0.813455   |

#### UB
| Train/Test | FER       | AV       | AU         |
|------------|-----------|----------|------------|
| FER        | 97.671638 | N/A      | N/A        |
| AV         | 94.126745 | 0.924197 | N/A        |
| AU         | 93.832434 | 0.915731 | 0.834955   |

## Results With Causality
### a) For task order: AU AV FER

#### LB
| Train/Test | AU       | AV       | FER        |
|------------|----------|----------|------------|
| AU         | 0.792395 | N/A      | N/A        |
| AV         | 0.820482 | 0.893078 | N/A        |
| FER        | 0.694927 | 0.101796 | 97.187754  |

#### EWC
| Train/Test | AU       | AV       | FER        |
|------------|----------|----------|------------|
| AU         | 0.849953 | N/A      | N/A        |
| AV         | 0.816259 | 0.867406 | N/A        |
| FER        | 0.665038 | 0.119556 | 97.207938  |

#### MAS
| Train/Test | AU       | AV       | FER        |
|------------|----------|----------|------------|
| AU         | 0.810051 | N/A      | N/A        |
| AV         | 0.820329 | 0.904089 | N/A        |
| FER        | 0.740221 | 0.207586 | 97.063121  |

#### SI
| Train/Test | AU       | AV       | FER        |
|------------|----------|----------|------------|
| AU         | 0.804695 | N/A      | N/A        |
| AV         | 0.815785 | 0.897547 | N/A        |
| FER        | 0.721289 | 0.195796 | 97.226329  |

#### NR
| Train/Test | AU       | AV       | FER        |
|------------|----------|----------|------------|
| AU         | 0.803792 | N/A      | N/A        |
| AV         | 0.877594 | 0.888073 | N/A        |
| FER        | 0.843339 | 0.679381 | 78.006511  |

#### UB
| Train/Test | AU       | AV       | FER        |
|------------|----------|----------|------------|
| AU         | 0.834487 | N/A      | N/A        |
| AV         | 0.871023 | 0.848828 | N/A        |
| FER        | 0.825194 | 0.734807 | 78.329497  |

### b) For task order: FER AV AU

#### LB
| Train/Test | FER       | AV       | AU         |
|------------|-----------|----------|------------|
| FER        | 97.383087 | N/A      | N/A        |
| AV         | 34.147788 | 0.926350 | N/A        |
| AU         | 3.238422  | 0.043446 | 0.814334   |

#### EWC
| Train/Test | FER       | AV       | AU         |
|------------|-----------|----------|------------|
| FER        | 97.226547 | N/A      | N/A        |
| AV         | 29.254031 | 0.914292 | N/A        |
| AU         | 4.649211  | 0.047790 | 0.828121   |

#### MAS
| Train/Test | FER       | AV       | AU         |
|------------|-----------|----------|------------|
| FER        | 97.559901 | N/A      | N/A        |
| AV         | 49.208752 | 0.905866 | N/A        |
| AU         | 23.158285 | 0.069486 | 0.861747   |

#### SI
| Train/Test | FER       | AV       | AU         |
|------------|-----------|----------|------------|
| FER        | 97.370429 | N/A      | N/A        |
| AV         | 41.464658 | 0.913485 | N/A        |
| AU         | 28.413661 | 0.301036 | 0.868619   |

#### NR
| Train/Test | FER       | AV       | AU         |
|------------|-----------|----------|------------|
| FER        | 97.809402 | N/A      | N/A        |
| AV         | 56.342128 | 0.912397 | N/A        |
| AU         | 24.036467 | 0.431922 | 0.822894   |

#### UB
| Train/Test | FER       | AV       | AU         |
|------------|-----------|----------|------------|
| FER        | 97.658314 | N/A      | N/A        |
| AV         | 95.222216 | 0.926570 | N/A        |
| AU         | 94.663628 | 0.923546 | 0.840273   |
