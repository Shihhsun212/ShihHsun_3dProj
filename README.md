-#Setup
##Feature Extraction → Pair Generation → Filtering (Doppelgangers++) → Feature Matching → Reconstruction

###1. Install Dependencies
###2. Follow setup instructions:
Doppelgangers++: https://github.com/doppelgangers25/doppelgangers-plusplus
HLoc: https://github.com/cvg/Hierarchical-Localization

your_project/
├── dataset/
│   └── your_images/
├── doppelgangers-plusplus/
│   ├── checkpoints/
│   │   └── checkpoint-dg+visym.pth
│   ├── filter_pairs.py          # Add this script
│   └── ...
└── Hierarchical-Localization/
    └── ...
    
###3.Download checkpoint-dg+visym.pth and place in(this is the pretrained model):
doppelgangers-plusplus/checkpoints/checkpoint-dg+visym.pth

###4.downlaod filter_pairs.py and place in
doppelgangers-plusplus/

#Usage
Navigate to HLoc directory:

