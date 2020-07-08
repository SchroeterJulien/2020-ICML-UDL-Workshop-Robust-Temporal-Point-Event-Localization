# Robust Temporal Point Event Localization through Smoothing and Counting
ICML 2020 Workshop on Uncertainty & Robustness in Deep Learning 

<div style="text-align: justify">
This work addresses the long-standing problem of robustly learning precise temporal event localization despite only having access to poorly aligned labels for training. To that end, we introduce a novel loss function that relaxes the reliance of the training on the exact position of labels, thus allowing for a softer learning of event localization. We demonstrate state-of-the-art performance against standard benchmarks in challenging experiments.
</div>

---
### (Section 4.1) Golf Swing Sequencing
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[Benchmark Code and Dataset]](https://github.com/wmcnally/golfdb) by McNally et al.

To run the **original code (CE)** with noise level (n) on split (s), (default: n=0, s=1):
```
python original_train.py $n $s
python original_eval.py $n $s
```

To run either the **classical one-sided-smoothing** (classic=True) or our **SoftLoc approach** (classic=False) with noise level (n) on split (s), (default: n=0, s=1, SoftLoc loss):
```
python soft_train.py $n $s $classic
python soft_eval.py $n $s $classic
```

The results are then save in .txt file in /results.

For the **causal** experiments, manually change the bidirectional argument in the EventDetect defintion to False (soft_train.py (l.69), soft_eval.py (l.92), original_train.py (l.42), and original_eval.py (l.80)).  


### (Section 4.2) Piano Onset Experiment
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[Benchmark Code]](https://github.com/tensorflow/magenta/tree/9885adef56d134763a89de5584f7aa18ca7d53b6) by Hawthorne et al.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[Dataset Request]](http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/) for MAPS Database.

To run the **SoftLoc** pipeline (to modify the noise level, change value in bash):
```
bash piepline.sh
```
Modify the oneSided variable in pipeline.sh in order to run the **one-sided smoothing benchmark**.

The project is structured as follows:

- pipeline.sh (Full pipeline)
- SoftNetworkModel.py (Tensorflow model with SoftLoc loss)
- main.py (Main script that runs the **training**)
- createDataset.py and google_create_dataset.py (**dataset** creation)
- infer.py and final_score.py (**inference**)
- config.py (Configuration file)

In addition, *subfolders* contains all utility functions used throughout the project.
