#!/usr/bin/env python
# jointly_train.py  –  GraphRAG-PD joint training + periodic validation
# ---------------------------------------------------------------
import os, json, torch, random, numpy as np
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time

from GraphRAG_train import EnhancedBERTScoreFeedbackRAG

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None,
                    help="path to checkpoint to resume from")
args = parser.parse_args()


# ------------ paths & device --------------------------------------------------
TRAIN_PATH = "/scratch1/tzhang62/GraphRAG_PD/Personalized_RAG/data/LAPDOG_new_dataset/train_data_10000.jsonl"
VALID_PATH = "/scratch1/tzhang62/GraphRAG_PD/Personalized_RAG/data/LAPDOG_new_dataset/valid_data_250.jsonl"
STORY_PATH = "/scratch1/tzhang62/GraphRAG_PD/Personalized_RAG/data/story.jsonl"
CLS_PATH   = "/scratch1/tzhang62/GraphRAG_PD/Personalized_RAG/model/classifer/consistency_classifier_fold5.pt"

CHECK_DIR  = "graphrag_model"
os.makedirs(CHECK_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"👉 training on: {device}")

# ------------ helpers ---------------------------------------------------------
def load_jsonl(path, cap=None):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if cap and i >= cap: break
            data.append(json.loads(line))
    return data

def construct_graph(texts, sent_encoder, thr=0.7):
    emb = sent_encoder.encode(texts, convert_to_tensor=True)
    sim = cosine_similarity(emb.cpu().numpy())
    ei  = [[], []]
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            if sim[i, j] >= thr:
                ei[0] += [i, j]
                ei[1] += [j, i]
    return Data(x=torch.tensor(emb.cpu().numpy(), dtype=torch.float),
                edge_index=torch.tensor(ei, dtype=torch.long))

def extract_persona(txt):
    if "context:" not in txt: return []
    p = txt.split("context:")[0].replace("persona:", "").strip()
    return [s.strip() for s in p.split('.') if s.strip()]

def extract_context(txt, keep=6):
    if "context:" not in txt: return []
    ctx = txt.split("context:")[1]
    ctx = ctx.replace("Q:", "Q: ").replace("R:", "R: ")
    pieces, out = [c for c in ctx.split("Q: ") if c.strip()], []
    for seg in pieces:
        q, *rest = seg.split("R: ")
        out.append(f"Q: {q.strip()}")
        if rest: out.append(f"R: {rest[0].strip()}")
    return out[-keep:]

# ------------ load base data --------------------------------------------------
train_data = load_jsonl(TRAIN_PATH)
valid_data = load_jsonl(VALID_PATH, cap=20)
story_data = load_jsonl(STORY_PATH)

sent_model = SentenceTransformer("Lajavaness/bilingual-embedding-small", device=device, trust_remote_code=True)
tokenizer  = AutoTokenizer.from_pretrained("t5-small")

# story graphs
print("📐 Building story graphs …")
story_graphs, story_sents = [], []
for st in tqdm(story_data):
    sents = [s.strip()+'.' for s in st['text'].split('. ') if s.strip()]
    story_sents.append(sents)
    if len(sents) > 1:
        story_graphs.append(construct_graph(sents, sent_model))
print(f"✅ {len(story_graphs)} story graphs ready.")

# ------------ dataset ---------------------------------------------------------
class PDSDataset(Dataset):
    def __init__(self, raw):
        self.raw = raw
        self.persona_graphs, self.persona_txt = [], []
        print("📐 Building persona graphs …")
        for item in tqdm(raw):
            pers = extract_persona(item['question'])
            self.persona_txt.append(pers)
            self.persona_graphs.append(
                construct_graph(pers if len(pers) > 1 else pers*2, sent_model)
            )

    def __len__(self): return len(self.raw)

    def __getitem__(self, idx):
        d  = self.raw[idx]
        ctx = extract_context(d['question'])
        enc = tokenizer(' '.join(ctx), padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        tgt = tokenizer(d['answers'][0], padding='max_length', truncation=True, max_length=512, return_tensors='pt').input_ids.squeeze()

        # crude similarity for contrastive positive
        pers_emb = sent_model.encode(' '.join(self.persona_txt[idx]), convert_to_tensor=True).cpu()
        best, best_i = -1, 0
        for i, sg in enumerate(story_graphs):
            sim = torch.cosine_similarity(pers_emb.cpu(), torch.mean(sg.x.cpu(), 0, keepdim=True)).item()
            if sim > best: best, best_i = sim, i

        return {
            "persona_graph":  self.persona_graphs[idx],
            "persona_sents":  self.persona_txt[idx],
            "input_ids":      enc.input_ids.squeeze(),
            "attn_mask":      enc.attention_mask.squeeze(),
            "labels":         tgt,
            "reference":      d['answers'][0],
            "positive_idx":   best_i,
            "context":        ctx
        }

train_set, valid_set = PDSDataset(train_data), PDSDataset(valid_data)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=lambda x: x)
valid_loader = DataLoader(valid_set, batch_size=4, shuffle=False, collate_fn=lambda x: x)

# ------------ model -----------------------------------------------------------
model     = EnhancedBERTScoreFeedbackRAG(hidden_dim=128, out_dim=64, classifier_path=CLS_PATH).to(device)
optimiser = optim.AdamW(model.parameters(), lr=5e-5)

# ------------ validation ------------------------------------------------------
@torch.no_grad()
def validate():
    model.eval()
    tot, tot_gen, tot_ret, tot_bs, n = 0.0, 0.0, 0.0, 0.0, 0
    samples = []

    for batch in valid_loader:
        for ex in batch:
            # sample K random stories + ensure the positive one is included
            K = 30
            sample_idx = np.random.choice(len(story_graphs), K, replace=False)
            if ex['positive_idx'] not in sample_idx:
                sample_idx[0] = ex['positive_idx']          # overwrite first slot

            pos_in_sample = int(np.where(sample_idx == ex['positive_idx'])[0][0])

            sample_graphs = [story_graphs[i].to(device) for i in sample_idx]
            sample_sents  = [story_sents[i]            for i in sample_idx]
            out = model(
                persona_graph      = ex['persona_graph'].to(device),
                story_graphs       =  sample_graphs,
                positive_idx       = pos_in_sample,
                input_ids          = ex['input_ids'].unsqueeze(0).to(device),
                attention_mask     = ex['attn_mask'].unsqueeze(0).to(device),
                labels             = ex['labels'].unsqueeze(0).to(device),
                references         = [ex['reference']],
                persona_sentences  = ex['persona_sents'],
                story_sentences_list = sample_sents
            )

            tot      += out['loss'].item()
            tot_gen  += out['generation_loss'].item()
            tot_ret  += (out['retrieval_loss'] or torch.tensor(0.)).item()
            tot_bs   += (out['bertscore_loss'] or torch.tensor(0.)).item()
            n += 1

            if len(samples) < 5:
                gen_ids = model.generator.generate(
                    input_ids=ex['input_ids'].unsqueeze(0).to(device),
                    attention_mask=ex['attn_mask'].unsqueeze(0).to(device),
                    encoder_outputs=out['encoder_outputs'],
                    max_length=100
                )
                samples.append(tokenizer.decode(gen_ids[0], skip_special_tokens=True))

    model.train()
    avg_tot, avg_gen = tot/n, tot_gen/n
    avg_ret, avg_bs  = tot_ret/n, tot_bs/n

    log_str  = (
    "\n--- Validation losses ----------------\n"
    f"total={avg_tot:.4f} | gen={avg_gen:.4f} | "
    f"retr={avg_ret:.4f} | bert={avg_bs:.4f}\n"
    "--------------------------------------\n\n"
    "--- Sample generations ---------------\n"
    )

    for i, s in enumerate(samples, 1):
        log_str += f"[{i}] {s}\n"
    log_str += "--------------------------------------\n\n"

    # 1️⃣  print to console
    print(log_str, end="")

    # 2️⃣  append to a file
    Path("/scratch1/tzhang62/GraphRAG_PD/Personalized_RAG/results/val_logs").mkdir(parents=True, exist_ok=True)
    fname = f"/scratch1/tzhang62/GraphRAG_PD/Personalized_RAG/results/val_logs/val_{time.strftime('%Y%m%d')}.txt"
    with open(fname, "a", encoding="utf-8") as f:
        f.write(log_str)

    return avg_tot


# ------------ training loop ---------------------------------------------------
num_epochs, eval_every = 10, 100
global_step, best_val = 0, float('inf')

if args.resume:
    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimiser.load_state_dict(ckpt["optimizer_state_dict"])
    global_step = ckpt.get("step", 0)
    start_epoch = ckpt.get("epoch", 1) + 1      # continue with next epoch
    best_val    = ckpt.get("val_loss", float("inf"))
    print(f"🔄  Resumed from {args.resume} | epoch {start_epoch} | "
          f"step {global_step} | best_val {best_val:.4f}")


print("🚀 Starting training …")
for epoch in range(start_epoch, num_epochs+1):
    prog = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in prog:
        for ex in batch:
            # sample K random stories + ensure the positive one is included
            K = 30
            sample_idx = np.random.choice(len(story_graphs), K, replace=False)
            if ex['positive_idx'] not in sample_idx:
                sample_idx[0] = ex['positive_idx']          # overwrite first slot

            pos_in_sample = int(np.where(sample_idx == ex['positive_idx'])[0][0])

            sample_graphs = [story_graphs[i].to(device) for i in sample_idx]
            sample_sents  = [story_sents[i]            for i in sample_idx]
            out = model(
                persona_graph      = ex['persona_graph'].to(device),
                story_graphs       = sample_graphs,
                positive_idx       = pos_in_sample,
                input_ids          = ex['input_ids'].unsqueeze(0).to(device),
                attention_mask     = ex['attn_mask'].unsqueeze(0).to(device),
                labels             = ex['labels'].unsqueeze(0).to(device),
                references         = [ex['reference']],
                persona_sentences  = ex['persona_sents'],
                story_sentences_list = sample_sents
            )
            loss      = out['loss']
            total_loss = out['loss'].item()
            gen_loss        = out['generation_loss'].item()
            retr_loss       = (out['retrieval_loss'] or torch.tensor(0.)).item()* model.lambda_retrieval
            bs_loss         = (out['bertscore_loss'] or torch.tensor(0.)).item()* model.lambda_bertscore

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        global_step += 1
        prog.set_postfix(loss=f"{loss.item():.4f}")

        if global_step % 10 == 0:
            print(f"\nStep {global_step}: "
                f"total={total_loss:.4f} | "
                f"gen={gen_loss:.4f} | "
                f"retr={retr_loss:.4f} | "
                f"bert={bs_loss:.4f} | "
                f"calc={gen_loss+retr_loss+bs_loss}\n")

        if global_step % eval_every == 0:
            val_loss = validate()
            print(f"🔎  step {global_step}: val_loss = {val_loss:.4f} (best {best_val:.4f})")

            if val_loss < best_val:
                best_val = val_loss
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                tag  = f"{val_loss:.4f}".replace('.', 'p')
                ckpt = f"{CHECK_DIR}/best_epoch_a100_{epoch}_step{global_step}_{tag}_{ts}.pt"
                torch.save({
                    "epoch": epoch,
                    "step":  global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimiser.state_dict(),
                    "val_loss": val_loss
                }, ckpt)
                print(f"✅  New best model saved → {ckpt}\n")

print("🎉 Training complete.")
