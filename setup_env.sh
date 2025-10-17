conda create -n openemma python=3.8
conda activate openemma

conda install nvidia/label/cuda-12.4.0::cuda-toolkit
pip install -r requirements.txt