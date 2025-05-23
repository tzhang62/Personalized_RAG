NGPU=1 CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=28888 --nproc_per_node=1 evaluate.py \
    --reader_model_type=google/t5-xl-lm-adapt \
    --text_maxlength=512 \
    --checkpoint_dir=ckpt/xl/eval \
    --model_path=ckpt/xl/ckpt/checkpoint/pth/ \
    --per_gpu_batch_size=1 \
    --eval_data="data/convai2/valid.jsonl" \
    --n_context=6 \
    --retriever_n_context=6 \
    --index_mode="flat" \
    --precision=fp32     \
    --save_index_path=ckpt/xl/eval \
    --write_results \
    --passages="data/corpora/story/story.jsonl" \
    --retriever_from=persona \
    --passages="data/corpora/story/story.jsonl" \
    --generation_num_beams=1 \
    --generation_length_penalty=1