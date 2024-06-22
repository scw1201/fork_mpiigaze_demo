# [FORK] A demo program of gaze estimation models (MPIIGaze, MPIIFaceGaze, ETH-XGaze)

[![PyPI version](https://badge.fury.io/py/ptgaze.svg)](https://pypi.org/project/ptgaze/)
[![Downloads](https://pepy.tech/badge/ptgaze)](https://pepy.tech/project/ptgaze)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hysts/pytorch_mpiigaze_demo/blob/master/demo.ipynb)
[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/hysts/pytorch_mpiigaze_demo.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/hysts/pytorch_mpiigaze_demo)

With this program, you can run gaze estimation on images and videos.
By default, the video from a webcam will be used.

![ETH-XGaze video01 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/eth-xgaze_video01.gif)
![ETH-XGaze video02 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/eth-xgaze_video02.gif)
![ETH-XGaze video03 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/eth-xgaze_video03.gif)

![MPIIGaze video00 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/mpiigaze_video00.gif)
![MPIIFaceGaze video00 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/mpiifacegaze_video00.gif)

![MPIIGaze image00 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/mpiigaze_image00.jpg)

To train a model for MPIIGaze and MPIIFaceGaze,
use [this repository](https://github.com/hysts/pytorch_mpiigaze).
You can also use [this repo](https://github.com/hysts/pl_gaze_estimation)
to train a model with ETH-XGaze dataset.

## Quick start

This program is tested only on Ubuntu.

### Installation

```bash
pip install ptgaze
```
PS: This command means that you download the configs files in `path-to-your-environment/lib/python3.8/site-packages/ptgaze/..`
    Modify them in the correct path, not the ones in this repository.
    
### Run demo

```bash
ptgaze --mode eth-xgaze
```

### Batch process

```bash
python batch_ptgaze.py --mode mpiigaze --folder assets/inputs
```
Added  batch_ptgaze.py & batch_main.py to develop the function of batch process

### Resolve the errors

##### 1. Download pretrained models
Models can be downloaded from this page https://github.com/hysts/pytorch_mpiigaze/issues/56.   (see `utils.py`'s downloading functions)

##### 2. NumPy error 
Due to NumPy's upgrade, need to modify codes about int64 & float64 in `path-to-your-environment/lib/python3.8/site-packages/ptgaze/..` if it runs into a problem :
   
```text
AttributeError: module 'numpy' has no attribute 'float'.
np.float was a deprecated alias for the builtin float. 
To avoid this error in existing code, use float by itself. 
Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use np.float64 here.
```
Modify:
```python
eg:
dtype=np.float ==> dtype=np.float64
np.int ==> np.int64
```
Modify all the error codes until it stops raising errors.

##### 3.torchvision: No url attribute error 

```text
error:"AttributeError: module 'torchvision.models.resnet' has no attribute 'model_urls'"
```
Modify `ptgaze/models/mpiifacegaze/backbones/resnet_simple.py`

from
```python
if pretrained_name:
    state_dict = torch.hub.load_state_dict_from_url(
        torchvision.models.resnet.model_urls[pretrained_name])
    self.load_state_dict(state_dict, strict=False)
```
to
```python
import torchvision.models as models
if pretrained_name:
    model_func = getattr(models, pretrained_name)
    pretrained_model = model_func(pretrained=True)
    self.load_state_dict(pretrained_model.state_dict(), strict=False)
```

### Usage


```
usage: ptgaze [-h] [--config CONFIG] [--mode {mpiigaze,mpiifacegaze,eth-xgaze}]
              [--face-detector {dlib,face_alignment_dlib,face_alignment_sfd,mediapipe}]
              [--device {cpu,cuda}] [--image IMAGE] [--video VIDEO] [--camera CAMERA]
              [--output-dir OUTPUT_DIR] [--ext {avi,mp4}] [--no-screen] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Config file. When using a config file, all the other commandline arguments
                        are ignored. See
                        https://github.com/hysts/pytorch_mpiigaze_demo/ptgaze/data/configs/eth-
                        xgaze.yaml
  --mode {mpiigaze,mpiifacegaze,eth-xgaze}
                        With 'mpiigaze', MPIIGaze model will be used. With 'mpiifacegaze',
                        MPIIFaceGaze model will be used. With 'eth-xgaze', ETH-XGaze model will be
                        used.
  --face-detector {dlib,face_alignment_dlib,face_alignment_sfd,mediapipe}
                        The method used to detect faces and find face landmarks (default:
                        'mediapipe')
  --device {cpu,cuda}   Device used for model inference.
  --image IMAGE         Path to an input image file.
  --video VIDEO         Path to an input video file.
  --camera CAMERA       Camera calibration file. See https://github.com/hysts/pytorch_mpiigaze_demo/
                        ptgaze/data/calib/sample_params.yaml
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        If specified, the overlaid video will be saved to this directory.
  --ext {avi,mp4}, -e {avi,mp4}
                        Output video file extension.
  --no-screen           If specified, the video is not displayed on screen, and saved to the output
                        directory.
  --debug
```

While processing an image or video, press the following keys on the window
to show or hide intermediate results:

- `l`: landmarks
- `h`: head pose
- `t`: projected points of 3D face model
- `b`: face bounding box


## References

- Zhang, Xucong, Seonwook Park, Thabo Beeler, Derek Bradley, Siyu Tang, and Otmar Hilliges. "ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation." In European Conference on Computer Vision (ECCV), 2020. [arXiv:2007.15837](https://arxiv.org/abs/2007.15837), [Project Page](https://ait.ethz.ch/projects/2020/ETH-XGaze/), [GitHub](https://github.com/xucong-zhang/ETH-XGaze)
- Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "Appearance-based Gaze Estimation in the Wild." Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. [arXiv:1504.02863](https://arxiv.org/abs/1504.02863), [Project Page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/)
- Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation." Proc. of the IEEE Conference on Computer Vision and Pattern Recognition Workshops(CVPRW), 2017. [arXiv:1611.08860](https://arxiv.org/abs/1611.08860), [Project Page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation/)
- Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation." IEEE transactions on pattern analysis and machine intelligence 41 (2017). [arXiv:1711.09017](https://arxiv.org/abs/1711.09017)
- Zhang, Xucong, Yusuke Sugano, and Andreas Bulling. "Evaluation of Appearance-Based Methods and Implications for Gaze-Based Applications." Proc. ACM SIGCHI Conference on Human Factors in Computing Systems (CHI), 2019. [arXiv](https://arxiv.org/abs/1901.10906), [code](https://git.hcics.simtech.uni-stuttgart.de/public-projects/opengaze)
