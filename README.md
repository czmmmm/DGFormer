# DGFormer: Dynamic Graph Transformer for 3D Human Pose Estimation

### Environment

The code is developed and tested under the following environment
* Python 3.8.8
* Pytorch 1.8.1
* CUDA 11.0

## Dataset
The Human3.6M dataset setting follows the [VideoPose3D](https://github.com/facebookresearch/VideoPose3D). Please refer to it to set up the Human3.6M dataset  (./data directory)

## Evaluation

We provide the pre-trained model (Human3.6M GT 2D poses as inputs) [here](https://drive.google.com/file/d/1UpUXwCwva5a9BNVSrpKp4AnIokWOf0hq/view?usp=share_link). Please put it into the `./checkpoint` directory and run as bellow:
```bash
python evaluate.py -k gt -c checkpoint --evaluate ckpt_h36m_gt_best_model.pth
```

## Acknowledgement

Part of our code is borrowed from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) and [GraFormer](https://github.com/Graformer/GraFormer). We thank the authors for releasing the codes.
