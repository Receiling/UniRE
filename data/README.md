## Dataset Processing

### ACE2004([https://catalog.ldc.upenn.edu/LDC2005T09](https://catalog.ldc.upenn.edu/LDC2005T09))

Firstly, achieving **ACE2004** dataset with scripts provided by [two-are-better-than-one](https://github.com/LorrinWWW/two-are-better-than-one/tree/master/datasets).
Secondly, splitting out the development set from the train set by [`split.py`](https://github.com/Receiling/UniRE/tree/master/data/split.py), and constructing 5-fold cross-validation data, i.e., fold{i} (i=1,2,3,4,5).
Lastly, getting the final data for each fold by running [`ace2004.sh`](https://github.com/Receiling/UniRE/tree/master/data/ace2004.sh).
```bash
./ace2004.sh ace2004_folder
```
Note that this script only processes fold1, which can be easily extended for all folds.

### ACE2005([https://catalog.ldc.upenn.edu/LDC2006T06](https://catalog.ldc.upenn.edu/LDC2006T06))

Firstly, achieving **ACE2005** dataset with scripts provided by [two-are-better-than-one](https://github.com/LorrinWWW/two-are-better-than-one/tree/master/datasets).
Then, getting the final data by running [`ace2005.sh`](https://github.com/Receiling/UniRE/tree/master/data/ace2005.sh)
```bash
./ace2005.sh ace2005_folder
```

### SciERC([http://nlp.cs.washington.edu/sciIE/](http://nlp.cs.washington.edu/sciIE/))

Firstly, downloading **SciERC** dataset from [sciIE](http://nlp.cs.washington.edu/sciIE/).
Then, getting the final data by running [`scierc.sh`](https://github.com/Receiling/UniRE/tree/master/data/scierc.sh)
```bash
./scierc.sh scierc_folder
```


