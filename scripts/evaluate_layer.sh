
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ALLOW_CODE_EVAL="1"

python modepd/eval.py --hf_model exp/layer/DeepSeek-V2-Lite-Chat-layer-pruned-1 \
    --tasks humaneval --num_fewshots 0 \
    --batch_size 8 --trust_remote_code --output_dir exp/layer/

python modepd/eval.py --hf_model exp/layer/DeepSeek-V2-Lite-Chat-layer-pruned-2 \
    --tasks humaneval cmmlu --num_fewshots 0 5 \
    --batch_size 8 --trust_remote_code --output_dir exp/layer/

python modepd/eval.py --hf_model exp/layer/DeepSeek-V2-Lite-Chat-layer-pruned-4 \
    --batch_size 8 --trust_remote_code --output_dir exp/layer/

python modepd/eval.py --hf_model exp/layer/DeepSeek-V2-Lite-Chat-layer-pruned-8 \
    --batch_size 8 --trust_remote_code --output_dir exp/layer/

python modepd/eval.py --hf_model exp/layer/DeepSeek-V2-Lite-Chat-layer-pruned-12 \
    --batch_size 8 --trust_remote_code --output_dir exp/layer/
