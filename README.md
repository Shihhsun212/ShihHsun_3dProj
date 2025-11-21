
# Pipepine : Feature Extraction → Pair Generation → Filtering (Doppelgangers++) → Feature Matching → Reconstruction
# Setup

![Screenshot of sparse cloud of KITP without using dp++](/comp.png)


### 1. Install Dependencies
### Follow setup instructions:
Doppelgangers++: https://github.com/doppelgangers25/doppelgangers-plusplus  
>git clone --recursive https://github.com/doppelgangers25/doppelgangers-plusplus.git 
cd doppelgangers-plusplus. 

>conda create -n doppelgangers_pp python=3.11 cmake=3.14.0. 
conda activate doppelgangers_pp  
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system  
pip install -r requirements.txt  
pip install -r   dust3r/requirements.txt   
pip install -r  dust3r/requirements_optional.txt  

HLoc: https://github.com/cvg/Hierarchical-Localization  
**Your folder should look like this(dont clone HLoc into DP++)**
>your_project/  
├── dataset/  
│   └── your_images/  
├── doppelgangers-plusplus/  
│   ├── checkpoints/  
│   │   └── checkpoint-dg+visym.pth  
│   ├── filter_pairs.py          # Add this script  
│   └── ...  
└── Hierarchical-Localization/  
    └── ...  
    
### 2.Download checkpoint-dg+visym.pth from https://huggingface.co/doppelgangers25/doppelgangers_plusplus/tree/main and place in(this is the pretrained model) 
>doppelgangers-plusplus/checkpoints/checkpoint-dg+visym.pth

### 3.downlaod filter_pairs.py and place in
>doppelgangers-plusplus/

# Usage
## Navigate to HLoc directory:
## Step 1: Extract Features
 SuperPoint features  
>python -m hloc.extract_features \
  --image_dir ../dataset/your_images \
  --export_dir outputs \
  --conf superpoint_max

NetVLAD features (for retrieval)  
>python -m hloc.extract_features \
  --image_dir ../dataset/your_images \
  --export_dir outputs \
  --conf netvlad

## Pair Generation
>python -m hloc.pairs_from_retrieval \
  --descriptors outputs/global-feats-netvlad.h5 \
  --output outputs/pairs.txt \
  --num_matched 20

## Filtering with Dp++
>cd ../doppelgangers-plusplus  
python filter_pairs.py \
  --images ../dataset/your_images \
  --pairs ../Hierarchical-Localization/outputs/pairs.txt \
  --output ../Hierarchical-Localization/outputs/pairs-filtered.txt \
  --checkpoint checkpoints/checkpoint-dg+visym.pth \
  --threshold 0.8

## Step 4: Match Filtered Pairs
>cd ../Hierarchical-Localization\
python -m hloc.match_features \
  --pairs outputs/pairs-filtered.txt \
  --features feats-superpoint-n4096-rmax1600.h5 \
  --export_dir outputs \
  --conf superglue
## Step 5: Reconstruction

## Citing
@misc{xiangli2024doppelgangersimprovedvisualdisambiguation,
      title={Doppelgangers++: Improved Visual Disambiguation with Geometric 3D Features}, 
      author={Yuanbo Xiangli and Ruojin Cai and Hanyu Chen and Jeffrey Byrne and Noah Snavely},
      year={2024},
      eprint={2412.05826},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.05826}, 
}

@inproceedings{sarlin2019coarse,
  title     = {From Coarse to Fine: Robust Hierarchical Localization at Large Scale},
  author    = {Paul-Edouard Sarlin and
               Cesar Cadena and
               Roland Siegwart and
               Marcin Dymczyk},
  booktitle = {CVPR},
  year      = {2019}
}

@inproceedings{sarlin2020superglue,
  title     = {{SuperGlue}: Learning Feature Matching with Graph Neural Networks},
  author    = {Paul-Edouard Sarlin and
               Daniel DeTone and
               Tomasz Malisiewicz and
               Andrew Rabinovich},
  booktitle = {CVPR},
  year      = {2020},
}
