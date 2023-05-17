# Cognition Guided Human-Object Realtionship Detection
Official Pytorch Implementation of our paper [Cognition Guided Human-Object Relationship Detection](https://ieeexplore.ieee.org/document/10112623) publish on **IEEE Transactions on Image Processing (TIP) 2023**. We propose a Relation-Pose Transformer (**RPT**) to detect human-object relationships in video. **RPT** can detect human-object relationships in each frame.

![GitHub Logo](/figure/framework.png)

**About the code**
We run the code on a single RTX3090 for both training and testing. We borrowed some code from [Yang's repository](https://github.com/jwyang/faster-rcnn.pytorch) and [Zellers' repository](https://github.com/rowanz/neural-motifs).

## Usage
First, clone the repository:
```
git clone https://github.com/zztao6/RPT.git
```
We recommend creating the environment by the yaml file
```
conda env create -f env_RPT.yaml 
```
We borrow some compiled code for bbox operations.
```
cd lib/draw_rectangles
python setup.py build_ext --inplace
cd ..
cd fpn/box_intersections_cpu
python setup.py build_ext --inplace
```
For the object detector part, please follow the compilation from https://github.com/jwyang/faster-rcnn.pytorch
We provide a pretrained FasterRCNN model for Action Genome. Please download [here](https://drive.google.com/file/d/1-u930Pk0JYz3ivS6V_HNTM1D5AxmN5Bs/view?usp=sharing) and put it in 
```
fasterRCNN/models/faster_rcnn_ag.pth
```

## Dataset
We use the dataset [Action Genome](https://www.actiongenome.org/#download) to train/evaluate our method. Please process the downloaded dataset with the [Toolkit](https://github.com/JingweiJ/ActionGenome). The directories of the dataset should look like:
```
|-- action_genome
    |-- annotations   #gt annotations
    |-- frames        #sampled frames
    |-- videos        #original videos
```
In the experiments for SGCLS/SGDET, we only keep bounding boxes with short edges larger than 16 pixels. Please download the file [object_bbox_and_relationship_filtersmall.pkl](https://drive.google.com/file/d/19BkAwjCw5ByyGyZjFo174Oc3Ud56fkaT/view?usp=sharing) and put it in the ```dataloader```.

In the experiments, we only use the frames with face confidence larger than 0.9. Please download the file [video_list_face_0.9.pkl](https://drive.google.com/file/d/1kQx7l1SraeJEhYRf8d0_kOAZWTl6xYc2/view?usp=share_link) and put it to ```dataloader```.

In the experiments, we use the trained head pose estimator. Please download the file [6DRepNet_300W_LP_AFLW2000.pth](https://drive.google.com/file/d/1Q5YgB58MF4okudTC305cXaY-llPOssca/view?usp=share_link) and put it to ```dataloader```.


## Train Image-based RPT
You can train the **Image-based RPT** with train_mini_batch.py. We trained the model on a single RTX3090:
+ For PredCLS: 
```
bash train_image_based_RPT_predcls.sh 
```
+ For SGCLS: 
```
bash train_image_based_RPT_sgcls.sh 
```
+ For SGDET: 
```
bash train_image_based-RPT_sgdet.sh 
```

## Train Video-based RPT
You can train the **Video-based RPT** with train_mini_batch.py. We trained the model on a single RTX3090:
+ For PredCLS: 
```
bash train_video_based_RPT_predcls.sh 
```
+ For SGCLS: 
```
bash train_video_based_RPT_sgcls.sh 
```
+ For SGDET: 
```
bash train_video_based-RPT_sgdet.sh 
```

## Evaluation Image-based RPT

You can evaluate the **Image-based RPT** with test_mini_batch.py.
Note that you can put the trained model into ```ckpt``` and edit the test file.

+ For PredCLS: 

```
bash test_image_based_RPT_predcls.sh 
```

+ For SGCLS: 

```
bash test_image_based_RPT_sgcls.sh 
```

+ For SGDET:

```
bash test_image_based_RPT_sgdet.sh
```

## Evaluation Video-based RPT

You can evaluate the **Image-based RPT** with test_mini_batch.py. 
Note that you can put the trained model into ```ckpt``` and edit the test file.

+ For PredCLS: 

```
bash test_video_based_RPT_predcls.sh 
```

+ For SGCLS: 

```
bash test_video_based_RPT_sgcls.sh 

```

+ For SGDET: 

```
bash test_video_based_RPT_sgdet.sh 
```

## Citation
If our work is helpful for your research, please cite our publication:

```
@article{zeng2023cognition,
  title={Cognition Guided Human-Object Relationship Detection},
  author={Zeng, Zhitao and Dai, Pengwen and Zhang, Xuan and Zhang, Lei and Cao, Xiaochun},
  journal={IEEE Transactions on Image Processing},
  year={2023},
  volume={32},
  pages={2468--2480},
  publisher={IEEE}
}
```
## Help 
When you have any question/idea about the code/paper. Please comment in Github or send us Email. We will reply as soon as possible.
# RPT
