# Table of contents
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
<!-- code_chunk_output -->

* [Table of contents](#table-of-contents)
* [Xlearn Transfer Learning Library](#xlearn-transfer-learning-library)
	* [Setup](#setup)
	* [Usage](#usage)
		* [Quick Start](#quick-start)
	* [Methods](#methods)

<!-- /code_chunk_output -->

# Xlearn Transfer Learning Library
The Xlearn transfer learning library implements some tranfer learning or domain
adaption algorithms using TensorFlow. The script `main.py` is for both training
and evaluating the models. And it also contains some scripts for downloading and
preprocessing the data and the pretrained models.

**This library is still being heavily developed. Vast changes are expected
these days.**

## Setup
This library is Python 2&3 compatible. See `requirements.txt` for the detailed
dependencies. You may modify it if you want to use the gpu version of
TensorFlow.

## Usage
### Quick Start
Excute the following script can run Alexnet-based DAN models on the Office
dataset.

```sh
# Prepare the data
sh data/office/download.sh
# Download and process the pretrained mean
python download_mean_from_caffe.py
# Download and process the pretrained model
python download_model_from_caffe.py
# Train
python main.py
```

## Methods
**Currently it only supports DAN with Caffe alexnet base-model.**
