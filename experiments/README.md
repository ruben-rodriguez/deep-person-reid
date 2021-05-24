While executing `torchreid.py`:

### GPU Usage

```
watch -n 2 nvidia-smi
```


```
Thu Apr 29 15:34:47 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 465.24.02    Driver Version: 465.24.02    CUDA Version: 11.3     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| 38%   65C    P0   113W / 175W |   3226MiB /  4042MiB |     99%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A       443      G   /usr/lib/Xorg                     396MiB |
|    0   N/A  N/A       854      G   /usr/bin/kwin_x11                   1MiB |
|    0   N/A  N/A       910      G   /usr/bin/plasmashell               64MiB |
|    0   N/A  N/A      1469      G   ...AAAAAAAAA= --shared-files       48MiB |
|    0   N/A  N/A      4200      G   ...AAAAAAAA== --shared-files       73MiB |
|    0   N/A  N/A      4545      G   ...AAAAAAAAA= --shared-files       18MiB |
|    0   N/A  N/A     16267      C   python                           2601MiB |
+-----------------------------------------------------------------------------+
```

### CPU Usage

```
watch -n 2 sensors
```

```
Adapter: ISA adapter
Package id 0:  +74.0°C  (high = +80.0°C, crit = +98.0°C)
Core 0:        +72.0°C  (high = +80.0°C, crit = +98.0°C)
Core 1:        +74.0°C  (high = +80.0°C, crit = +98.0°C)
Core 2:        +70.0°C  (high = +80.0°C, crit = +98.0°C)
Core 3:        +70.0°C  (high = +80.0°C, crit = +98.0°C)
```


### Cross-domain ReID

Better performance by using random_flip + color_jitter transforms.
Transforms are available at 
- https://pytorch.org/vision/stable/transforms.html
- https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/data/transforms.py

Some general descriptions:
- random_flip: 


### Visualizing metrics

#### Learning curves 

```
tensorboard --logdir=log/resnet50
```

#### Ranking results

Use visrank.py

#### Activation maps

```
python tools/visualize_actmap.py \
--root $DATA/reid \
-d market1501 \
-m osnet_x1_0 \
--weights PATH_TO_PRETRAINED_WEIGHTS \
--save-dir log/visactmap_osnet_x1_0_market1501
```


### Metrics 

Following [45],
two metrics are used to evaluate person re-ID accuracy –
mean Average Precision (mAP), which is the mean across
all queries’ Average Precision (AP) and the rank-1, 10, 20
accuracy denoting the possibility to locate at least one true
positive in the top-1, 10, 20 ranks.


two metrics are used to evaluate person re-ID accuracy –
mean Average Precision (mAP), which is the mean across
all queries’ Average Precision (AP) and the rank-1, 10, 20
accuracy denoting the possibility to locate at least one true
positive in the top-1, 10, 20 ranks.


Precision measures how accurate is your predictions. i.e. the percentage of your predictions are correct. It is given as the ratio of true positive (TP) and the total number of predicted positives. 
Recall measures how good you find all the positives. is defined as the ratio of TP and total of ground truth positives

However, all these metrics fail when it comes to determining if a model is performing well in information retrieval or object detection tasks. This where mAP comes to the rescue!


Just by looking at the formulas, we could suspect that for a given classification model, there lies a trade-off between its precision and recall performance. 
If we are using a neural network, this trade-off can be adjusted by the model’s final layer soft-max threshold.

https://laptrinhx.com/breaking-down-mean-average-precision-map-2583130432/


https://lucasxlu.github.io/blog/2020/03/05/cv-reid/


Average of precision values at ranks of relevant results.
Heavy penalties for queries with low performance.

average precions qn = ()