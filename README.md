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
Filtering with filter_pairs(Please change the directories inside the code) 
Reconstruct the model using the filter_paired


    
