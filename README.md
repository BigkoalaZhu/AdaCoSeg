# AdaCoSeg
This is an Pytorch demo of the paper "[SCORES: Shape Composition with Recursive Substructure Priors](https://kevinkaixu.net/projects/scores.html)". This is a neural network which learns structure fusion for 3D shape composition

## Usage
**Dependancy**

This implementation should be run with Python 3.x and Pytorch 0.4.0.

We provide a demo dataset for training and testing, you can download it from this "[link](https://www.dropbox.com/s/tnyxvwlqul5feqo/chair.zip?dl=0)". Put the unziped files in the folder `/chair`.

**Demo**
You can train your own offline network through:
```
python trainOffline.py
```
Or, you can use our pretrained model `/chair/PartSpace_Training.pkl` to run two interesting demo.

1. Run cosegmentation on the testing dataset, the results would be saved in /coseg.
```
python demo_cosegmentation.py
```
2. Run cosegmentation on the training dataset, the results would be saved in /refineTraining. You can compare the segmentation consistency before and after the cosegmentation.
```
python demo_refineTrainingData 
```

## Citation
If you use this code, please cite the following paper.
```
@misc{zhu2019adacoseg,
    title={AdaCoSeg: Adaptive Shape Co-Segmentation with Group Consistency Loss},
    author={Chenyang Zhu and Kai Xu and Siddhartha Chaudhuri and Li Yi and Leonidas Guibas and Hao Zhang},
    year={2019},
    eprint={1903.10297},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
