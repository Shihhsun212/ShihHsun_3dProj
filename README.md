Pipeline
Matching Pair using HLoc -> Filtering with Dp++ -> Reconstruct the sparse model using fitered-pairs  

follow instruction on https://github.com/doppelgangers25/doppelgangers-plusplus set up enviroment   
clone https://github.com/cvg/Hierarchical-Localization  

Folder should ended up like this
|---data  
    |---doppelgangers-plusplus  
    |---Hierarchical-Localization  
    |---dataset  

Using the pretrained model  
  Download checkpoint-dg+visym.pth from https://huggingface.co/doppelgangers25/doppelgangers_plusplus/tree/main  
  place the file  
    |---data  
      |---doppelgangers-plusplus  
        |---checkpoints  

Place the filter_pairs(filtering code using Dp++) and place it under 
|---data  
    |---doppelgangers-plusplus  

Run feature matching using HLoc   
    using Netvlad(for retrieval) and superpoint(or any feature extracter)
    (Optional)Run matching for the unfiltered

Filtering with filter_pairs(Please change the directories inside the code) 
Run python filter_pairs.py 
      --images "C:\Users\shuxu\Desktop\recon3d\dataset\Kavli_Dataset" `  
      --pairs "..\Hierarchical-Localization\outputs\pairs.txt" `  
      --output "..\Hierarchical-Localization\outputs\pairs-filtered.txt" `  
      --checkpoint "checkpoints\checkpoint-dg+visym.pth" `  
      --threshold 0.8  

Run python -m hloc.match_features `
  --pairs outputs\pairs-filtered.txt `
  --features feats-superpoint-n4096-rmax1600.h5 `
  --export_dir outputs `
  --conf superglue

Reconstruct the model using the output



    
