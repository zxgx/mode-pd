
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ALLOW_CODE_EVAL="1"

python modepd/eval.py --hf_model exp/weight/DeepSeek-V2-Lite-Chat-weight-pruned-95 \
    --tasks humaneval --num_fewshots 0 \
    --batch_size 8 --trust_remote_code --output_dir exp/weight/

python modepd/eval.py --hf_model exp/weight/DeepSeek-V2-Lite-Chat-weight-pruned-90 \
    --tasks humaneval cmmlu --num_fewshots 0 5 \
    --batch_size 8 --trust_remote_code --output_dir exp/weight/

python modepd/eval.py --hf_model exp/weight/DeepSeek-V2-Lite-Chat-weight-pruned-85 \
    --batch_size 8 --trust_remote_code --output_dir exp/weight/

python modepd/eval.py --hf_model exp/weight/DeepSeek-V2-Lite-Chat-weight-pruned-80 \
    --batch_size 8 --trust_remote_code --output_dir exp/weight/
