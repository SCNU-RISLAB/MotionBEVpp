# open3d-semantickitti-mos-visualize
- visualize the semantickitti dataset's both moving and movable label

### How to use it
- Firstly, you need to install open3d == 0.16.0
- Change the param in viz.sh, including the **dataset path**、 **gt label path** and **pred label path**
- Start it as follows:
```shell
bash viz.sh
```

By the way, the structure of the pred label path is like the following:
```
└── predictions
    └── sequences
        ├── 00
        │   ├── predictions
        │   │    ├── 000000.label
        │   │    ├── 000001.label
        │   │    ├── 000002.label
        │   │    ├── 000003.label
        │   │    ├── 000004.label
        │   │    ├── ......
        │   │    └── xxxxxx.label        
        │   ├── predictions_fused
        │   │    ├── 000000.label
        │   │    ├── 000001.label
        │   │    ├── 000002.label
        │   │    ├── 000003.label
        │   │    ├── 000004.label
        │   │    ├── ......
        │   │    └── xxxxxx.label        
        │   ├── predictions_movable
        │   │    ├── 000000.label
        │   │    ├── 000001.label
        │   │    ├── 000002.label
        │   │    ├── 000003.label
        │   │    ├── 000004.label
        │   │    ├── ......
        │   │    └── xxxxxx.label
```