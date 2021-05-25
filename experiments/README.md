## Experiments

- OSNet training & test using market1501 (`osnet.py`)
- OSNet cross-evaluation, using market1501 as source, cuhk03 as target. (`cross-evaluation.py`)
- ResNet50 training & test using market1501 (`resnet.py`)

## Hardware Usage

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

### Visualizing metrics

#### Learning curves 

`tensorboard --logdir=log/resnet50`
`tensorboard --logdir=log/osnet_x1_0/`

#### Ranking results

Use visrank.py: `python visrank.py osnet_x1_0`

#### Activation maps

```
python tools/visualize_actmap.py \
--root $DATA/reid \
-d market1501 \
-m osnet_x1_0 \
--weights PATH_TO_PRETRAINED_WEIGHTS \
--save-dir log/visactmap_osnet_x1_0_market1501
```
