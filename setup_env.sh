python3 -m venv .venv
source .venv/bin/activate

pip install cuda-toolkit
pip install -r requirements.txt
pip install flash-attn  # only for gpu envs