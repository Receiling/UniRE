## Dataset Processing

### ACE2004

Firstly, achieving **ACE2004** dataset with scripts provided by [two-are-better-than-one](https://github.com/LorrinWWW/two-are-better-than-one/tree/master/datasets).
Secondly, splitting out the development set from the train set by [`split.py`](https://github.com/Receiling/UniRE/data/split.py), and constructing 5-fold cross-validation data, i.e., fold{i} (i=1,2,3,4,5).
Lastly, getting the final data for each fold by running [`ace2004.sh`](https://github.com/Receiling/UniRE/data/ace2004.sh).
```bash
./ace2004.sh ace2004_folder
```
Note that this script only processes fold1, which can be easily extended for all folds.

### ACE2005

Firstly, achieving **ACE2005** dataset with scripts provided by [two-are-better-than-one](https://github.com/LorrinWWW/two-are-better-than-one/tree/master/datasets).
Then, getting the final data by running [`ace2005.sh`](https://github.com/Receiling/UniRE/data/ace2005.sh)
```bash
./ace2005.sh ace2005_folder
```

### SciERC

Firstly, downloading **SciERC** dataset from [sciIE](http://nlp.cs.washington.edu/sciIE/).
Then, getting the final data by running [`scierc.sh`](https://github.com/Receiling/UniRE/data/scierc.sh)
```bash
./scierc.sh scierc_folder
```


