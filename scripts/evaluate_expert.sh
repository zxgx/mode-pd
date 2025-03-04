
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ALLOW_CODE_EVAL="1"

python modepd/eval.py --hf_model exp/expert/DeepSeek-V2-Lite-Chat-expert-pruned-56 \
    --tasks humaneval --num_fewshots 0 \
    --batch_size 8 --trust_remote_code --output_dir exp/expert/

python modepd/eval.py --hf_model exp/expert/DeepSeek-V2-Lite-Chat-expert-pruned-48 \
    --tasks humaneval cmmlu --num_fewshots 0 5 \
    --batch_size 8 --trust_remote_code --output_dir exp/expert/

python modepd/eval.py --hf_model exp/expert/DeepSeek-V2-Lite-Chat-expert-pruned-40 \
    --batch_size 8 --trust_remote_code --output_dir exp/expert/

python modepd/eval.py --hf_model exp/expert/DeepSeek-V2-Lite-Chat-expert-pruned-32 \
    --batch_size 8 --trust_remote_code --output_dir exp/expert/
