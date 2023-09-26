#!/bin/bash
#SBATCH --job-name=llama7b-2_chat_augmented_C_mmlu_cm_mg
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=6
#SBATCH --gres=gpu:1
#SBATCH --output=llama7b-2_chat_augmented_C_mmlu_cm_mg.%j.out
#SBATCH --partition=a40
#SBATCH --qos=normal

python main.py --model hf-causal --model_args pretrained=/model-weights/Llama-2-7b-chat-hf/,tokenizer=/ssd005/projects/llm/Llama-2-7b-chat-hf/ --tasks mmlu_college_med,mmlu_medical_genetics --device cuda:0 --no_cache --output_base_path /scratch/ssd004/scratch/ywchoi/ --write_out