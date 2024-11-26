
import faiss
import os
os.environ["MKL_NUM_THREADS"] = "4"
import re
import time
from collections import defaultdict

import numpy as np
import torch
import torch.cuda
import torch.distributed as dist

from src import dist_utils, slurm, util
from src.index_io import load_or_initialize_index, save_embeddings_and_index
from src.model_io import create_checkpoint_directories, load_or_initialize_lapdog_model
from src.options import get_options
from src.tasks import get_task
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "true"
#os.environ["MKL_NUM_THREADS"] = "4"

def _get_eval_data_iterator(opt, data_path, task):
    data_iterator = task.data_iterator(data_path, opt.global_rank, opt.world_size, opt=opt, is_eval=True)
    data_iterator = filter(None, map(task.process, data_iterator))
    data_iterator = list(task.batch_iterator(data_iterator, opt.per_gpu_eval_batch_size if opt.per_gpu_eval_batch_size > 0 else opt.per_gpu_batch_size))

    if dist.is_initialized():
        len_data = torch.tensor(len(data_iterator), device=torch.device("cuda"))
        dist.all_reduce(len_data, torch.distributed.ReduceOp.MAX)
        dist.barrier()
        if len(data_iterator) < len_data.item():
            data_iterator.extend([{} for _ in range(len_data.item() - len(data_iterator))])

    return data_iterator


@torch.no_grad()
def run_retrieval_only(model, index, opt, data_path, step=None):
    model.eval()
    metrics = defaultdict(lambda: [])
    dataset_wpred = []
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    reader_tokenizer = unwrapped_model.reader_tokenizer

    task = get_task(opt, reader_tokenizer)
    data_iterator = _get_eval_data_iterator(opt, data_path, task)
    for i, batch in tqdm(enumerate(data_iterator), total=len(data_iterator)):
        query = batch.get("query", [""])
        answers = batch.get("target", [""])
        batch_metadata = batch.get("metadata")
        query_enc = model.retriever_tokenize(query)
        if opt.retriever_from == 'persona':
            retrieval_query = [re.sub(".*persona:|.*persona for R:|context:.*| dialog:.*", '', q) for q in query]
            query_enc = model.retriever_tokenize(retrieval_query)
        retrieved_passages, _ = unwrapped_model.retrieve(
            index,
            opt.n_context,
            query,
            query_enc["input_ids"].cuda(),
            query_enc["attention_mask"].cuda(),
            batch_metadata=batch_metadata,
            filtering_fun=task.filter,
        )
        # If example is a padding example then skip step
        if (len(query) == 0) or (len(query[0]) == 0):
            continue
        for k in range(len(retrieved_passages)):
            if opt.write_results:
                gold = [answers[k]] if not "answers" in batch else batch["answers"][k]
                ex = {"query": query[k], "answers": gold, "passages": retrieved_passages[k]}
                if batch_metadata is not None:
                    ex["metadata"] = batch_metadata[k]
                if "id" in batch:
                    ex["id"] = batch["id"][k]
                dataset_wpred.append(ex)

    if opt.write_results:
        dataset_name, _ = os.path.splitext(os.path.basename(data_path))
        dataset_name = f"{dataset_name}-step-{step}"
        util.save_distributed_dataset(dataset_wpred, dataset_name, opt)

    return metrics


@torch.no_grad()
def evaluate(model, index, opt, data_path, step=None):
    model.eval()
    metrics = defaultdict(lambda: [])
    dataset_wpred = []
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    reader_tokenizer = unwrapped_model.reader_tokenizer

    task = get_task(opt, reader_tokenizer)
    data_iterator = _get_eval_data_iterator(opt, data_path, task)
    data_iterator = tqdm(enumerate(data_iterator), total=len(data_iterator))
    for i, batch in data_iterator:
        query = batch.get("query", [""])
        answers = batch.get("target", [""])
        batch_metadata = batch.get("metadata")
        target_tokens = batch.get("target_tokens")

        query_enc, labels, decoder_input_ids = unwrapped_model.tokenize(query, answers, target_tokens=target_tokens)
        retrieval_query = unwrapped_model.build_retrieval_query(query)
        retrieval_query_enc, _, _ = unwrapped_model.tokenize(retrieval_query, answers, target_tokens=target_tokens)
        if not opt.use_file_passages:
            query_ids_retriever = retrieval_query_enc["input_ids"].cuda()
            query_mask_retriever = retrieval_query_enc["attention_mask"].cuda()
            retrieved_passages, _ = unwrapped_model.retrieve(
                index,
                opt.n_context,
                query,
                query_ids_retriever,
                query_mask_retriever,
                batch_metadata=batch_metadata,
                filtering_fun=task.filter,
            )
        else:
            assert "passages" in batch, "cant use use_file_passages mode without passing in passages"
            retrieved_passages = [p[: opt.n_context] for p in batch["passages"]]

        # If example is a padding example then skip step
        if (len(query) == 0) or (len(query[0]) == 0):
            continue

        if opt.reader_causallm is not None:
            labels, reader_tokens, _ = unwrapped_model.tokenize_passages_causallm(query, retrieved_passages, answers)
        reader_tokens, _ = unwrapped_model.tokenize_passages(query, retrieved_passages)

        if "eval_loss" in task.metrics:
            eval_loss, logits = unwrapped_model.compute_reader_loss_and_logits(reader_tokens, decoder_input_ids, labels)
            metrics["eval_loss"].append(eval_loss)

        generation = unwrapped_model.generate(
            reader_tokens, query, choices=batch["choices"] if "choices" in batch else None
        )

        for k, g in enumerate(generation):
            if opt.decoder_prompt_format is not None:
                query_ids = reader_tokenizer.encode(
                    opt.decoder_prompt_format.format_map({"query": query[k]}), add_special_tokens=False
                )
                g = g[len(query_ids) + 1:]
            pred = reader_tokenizer.decode(g, skip_special_tokens=True)
            gold = [answers[k]] if not "answers" in batch else batch["answers"][k]
            sample_metrics = task.evaluation(pred, gold)
            for key, value in sample_metrics.items():
                metrics[key].append(value)

            if opt.write_results:
                ex = {"query": query[k], "answers": gold, "generation": pred}
                if not opt.dont_write_passages:
                    ex["passages"] = retrieved_passages[k]
                if batch_metadata is not None:
                    ex["metadata"] = batch_metadata[k]
                if opt.task == "multiple_choice":
                    ex["choice_logits"] = task.get_choice_logits(logits[k])
                if "id" in batch:
                    ex["id"] = batch["id"][k]
                print(ex)
                dataset_wpred.append(ex)

    metrics, dataset_wpred = task.evaluation_postprocessing(metrics, dataset_wpred)
    metrics = util.avg_dist_dict(task.metrics, metrics)
    metrics = {key: value if key == "eval_loss" else 100 * value for key, value in metrics.items()}
    if opt.write_results:
        dataset_name, _ = os.path.splitext(os.path.basename(data_path))
        dataset_name = f"{dataset_name}-step-{step}"
        util.save_distributed_dataset(dataset_wpred, dataset_name, opt)

    return metrics


# if __name__ == "__main__":
#     options = get_options()
#     opt = options.parse()

#     torch.manual_seed(opt.seed)
#     slurm.init_distributed_mode(opt)
#     slurm.init_signal_handler()

#     checkpoint_path, saved_index_path = create_checkpoint_directories(opt)

#     logger = util.init_logger(opt.is_main, opt.is_distributed, os.path.join(checkpoint_path, "run.log"))
#     if opt.is_main:
#         options.print_options(opt)

#     logger.info(f"world size: {dist_utils.get_world_size()}")

#     index, passages = load_or_initialize_index(opt)
#     model, _, _, _, _, opt, step = load_or_initialize_lapdog_model(opt, eval_only=True)

#     logger.info("Start Evaluation")
#     print("Start Evaluation")
#     dist_utils.barrier()

#     if not opt.use_file_passages and opt.load_index_path is None:
#         indexing_start = time.time()
#         model.build_index(index, passages, opt.per_gpu_embedder_batch_size, logger)

#         if opt.save_index_path is not None:
#             save_embeddings_and_index(index, opt)

#     for data_path in opt.eval_data:
#         dataset_name = os.path.basename(data_path)
#         logger.info(f"Start Evaluation on {data_path}")
#         if opt.retrieve_only:
#             run_retrieval_only(model, index, opt, data_path, step)
#         else:
#             metrics = evaluate(model, index, opt, data_path, step)
#             log_message = f"Dataset: {dataset_name}"
#             for k, v in metrics.items():
#                 log_message += f" | {v:.3f} {k}"
#             logger.info(log_message)

if __name__ == "__main__":
    options = get_options()
    opt = options.parse()

    # Set seed for reproducibility
    torch.manual_seed(opt.seed)

    # Set distributed flags manually (disable SLURM distributed mode)
    opt.is_distributed = False  # Set to True if you manually initialize distributed mode
    opt.is_main = True          # Set to True for a single process or if it's the main process in distributed
    
    if torch.cuda.is_available():
        opt.device = 'cuda'
    else:
        opt.device = 'cpu'
    
    opt.global_rank=0
    opt.world_size=1

    # Initialize distributed process group if needed
    if opt.is_distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="tcp://localhost:12355", world_size=1, rank=0)
        dist_utils.barrier()  # Synchronize if in a distributed environment

    checkpoint_path, saved_index_path = create_checkpoint_directories(opt)
    
    # Initialize logger without SLURM
    logger = util.init_logger(opt.is_main, opt.is_distributed, os.path.join(checkpoint_path, "run.log"))
    if opt.is_main:
        options.print_options(opt)

    # Log the world size (will be 1 if not distributed)
    logger.info(f"world size: {dist_utils.get_world_size() if opt.is_distributed else 1}")

    # Load index and model
    index, passages = load_or_initialize_index(opt)
    model, _, _, _, _, opt, step = load_or_initialize_lapdog_model(opt, eval_only=True)

    logger.info("Start Evaluation")
    print("Start Evaluation")
    if opt.is_distributed:
        dist_utils.barrier()

    # Build the index if needed
    if not opt.use_file_passages and opt.load_index_path is None:
        indexing_start = time.time()
        model.build_index(index, passages, opt.per_gpu_embedder_batch_size, logger)

        if opt.save_index_path is not None:
            save_embeddings_and_index(index, opt)

    # Evaluation loop
    for data_path in opt.eval_data:
        dataset_name = os.path.basename(data_path)
        logger.info(f"Start Evaluation on {data_path}")
        if opt.retrieve_only:
            run_retrieval_only(model, index, opt, data_path, step)
        else:
            metrics = evaluate(model, index, opt, data_path, step)
            log_message = f"Dataset: {dataset_name}"
            for k, v in metrics.items():
                log_message += f" | {v:.3f} {k}"
            logger.info(log_message)

    # Cleanup (if distributed mode was manually initialized)
    if opt.is_distributed:
        torch.distributed.destroy_process_group()
