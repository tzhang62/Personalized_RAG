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
from transformers import get_linear_schedule_with_warmup
from GraphRAG_train import EnhancedBERTScoreFeedbackRAG
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
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

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased").to("cuda" if torch.cuda.is_available() else "cpu")
bert_model.eval()

def compute_bertscore(story_text, reference_text):
    with torch.no_grad():
        inputs_s = bert_tokenizer(story_text, return_tensors="pt", truncation=True, padding=True).to(bert_model.device)
        inputs_r = bert_tokenizer(reference_text, return_tensors="pt", truncation=True, padding=True).to(bert_model.device)

        embed_s = bert_model(**inputs_s).last_hidden_state[0][inputs_s.attention_mask[0].bool()]
        embed_r = bert_model(**inputs_r).last_hidden_state[0][inputs_r.attention_mask[0].bool()]

        embed_s = F.normalize(embed_s, dim=-1)
        embed_r = F.normalize(embed_r, dim=-1)

        sim_matrix = embed_s @ embed_r.T
        prec = sim_matrix.max(dim=1).values.mean()
        rec = sim_matrix.max(dim=0).values.mean()
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        return f1.item()


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
story_data = load_jsonl(STORY_PATH, cap=20000)

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
    def __init__(self, raw, story_data):
        self.raw = raw
        self.persona_graphs, self.persona_txt = [], []
        print("📐 Building persona graphs …")
        for item in tqdm(raw):
            pers = extract_persona(item['question'])
            self.persona_txt.append(pers)
            self.persona_graphs.append(
                construct_graph(pers if len(pers) > 1 else pers*2, sent_model)
            )

        self.story_texts = []
        for item in tqdm(story_data):
            self.story_texts.append(item)


    def __len__(self): return len(self.raw)

    def __getitem__(self, idx):
        d  = self.raw[idx]
        reference = d['answers'][0]
        ctx = extract_context(d['question'])
        enc = tokenizer(' '.join(ctx), padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        tgt = tokenizer(d['answers'][0], padding='max_length', truncation=True, max_length=512, return_tensors='pt').input_ids.squeeze()
        best_score, best_idx = -1, 0
        for i, story_text in enumerate(self.story_texts[idx]):  # assuming you have raw text
            score = compute_bertscore(story_text, reference)
            if score > best_score:
                best_score, best_idx = score, i

        
        return {
            "persona_graph":  self.persona_graphs[idx],
            "persona_sents":  self.persona_txt[idx],
            "input_ids":      enc.input_ids.squeeze(),
            "attn_mask":      enc.attention_mask.squeeze(),
            "labels":         tgt,
            "reference":      d['answers'][0],
            "positive_idx":   best_idx,
            "context":        ctx
        }

class PDSDatasetvalid(Dataset):
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

train_set, valid_set = PDSDataset(train_data,story_data), PDSDatasetvalid(valid_data)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=lambda x: x)
valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False, collate_fn=lambda x: x)

# ------------ model -----------------------------------------------------------
model     = EnhancedBERTScoreFeedbackRAG(hidden_dim=128, out_dim=64, classifier_path=CLS_PATH).to(device)
optimiser = optim.AdamW(model.parameters(), lr=5e-5)
num_epochs, eval_every = 20, 5
# compute total number of training steps
steps_per_epoch = len(train_loader)
total_steps      = num_epochs * steps_per_epoch

# choose how many warmup steps you’d like (e.g. 10% of total)
warmup_steps = int(0.1 * total_steps)

# create the scheduler
scheduler = get_linear_schedule_with_warmup(
    optimiser,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)
# ------------ validation ------------------------------------------------------
@torch.no_grad()
def validate():
    model.eval()
    tot, tot_gen, tot_ret, tot_rl, tot_bs, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
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
            tot_rl   += (out['rl_loss'] or torch.tensor(0.)).item()
            n += 1

            if len(samples) < 10:
                gen_ids = model.generator.generate(
                    input_ids=ex['input_ids'].unsqueeze(0).to(device),
                    attention_mask=ex['attn_mask'].unsqueeze(0).to(device),
                    encoder_outputs=out['encoder_outputs'],
                    do_sample=True,
                    temperature=0.9,
                    max_length=60,
                    no_repeat_ngram_size=3,      # forbid repeating any 3-gram
                    repetition_penalty=1.2     
                )
                gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                bs = model.compute_bertscore_f1([gen_text], [ex['reference']])  # scalar tensor
                tot_bs += bs.item()
                samples.append((gen_text, ex['reference']))   # ← keep reference too

    model.train()
    avg_tot, avg_gen = tot/n, tot_gen/n
    avg_ret, avg_rl  = tot_ret/n, tot_rl/n
    avg_bs = tot_bs/10


    log_str  = (
    "\n--- Validation losses ----------------\n"
    f"total={avg_tot:.4f} | gen={avg_gen:.4f} | "
    f"retr={avg_ret:.4f} | rl={avg_rl:.4f} | bs={avg_bs:.4f}\n"
    "--------------------------------------\n\n"
    "--- Sample generations ---------------\n"
    )

    for i, (gen, ref) in enumerate(samples, 1):
        log_str += f"[{i}] GEN: {gen}\n"
        log_str += f"    REF: {ref}\n"               # ← add reference line
    log_str += "--------------------------------------\n\n"

    # 1️⃣  print to console
    print(log_str, end="")

    # 2️⃣  append to a file
    Path("/scratch1/tzhang62/GraphRAG_PD/Personalized_RAG/results/val_logs").mkdir(parents=True, exist_ok=True)
    fname = f"/scratch1/tzhang62/GraphRAG_PD/Personalized_RAG/results/val_logs/rl_val_{time.strftime('%Y%m%d')}.txt"
    with open(fname, "a", encoding="utf-8") as f:
        f.write(log_str)

    return avg_bs


# ------------ training loop ---------------------------------------------------

global_step, best_bs = 0, 0
start_epoch=1

if args.resume:
    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimiser.load_state_dict(ckpt["optimizer_state_dict"])
    global_step = ckpt.get("step", 0)
    start_epoch = ckpt.get("epoch", 1) + 1      # continue with next epoch
    
    best_bs = 0.5
    print(f"🔄  Resumed from {args.resume} | epoch {start_epoch} | "
          f"step {global_step} ")


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
            rl_loss         = (out['rl_loss'] or torch.tensor(0.)).item()* model.lambda_rl

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()

        global_step += 1
        prog.set_postfix(loss=f"{loss.item():.4f}")

        

        if global_step % 5 == 0:
            print(f"\nStep {global_step}: "
                f"total={total_loss:.4f} | "
                f"gen={gen_loss:.4f} | "
                f"retr={retr_loss:.4f} | "
                f"rl={rl_loss:.4f} | "
                f"calc={gen_loss+retr_loss+rl_loss}\n")

        if global_step % eval_every == 0:
            bs = validate()
            print(f"🔎  step {global_step}: bs = {bs:.4f} (best {best_bs:.4f})")
            if bs > best_bs:
            
                best_bs = bs
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                tag  = f"{bs:.4f}".replace('.', 'p')
                ckpt = f"{CHECK_DIR}/fast_best_epoch_check{epoch}_step{global_step}_{tag}_{ts}.pt"
                torch.save({
                    "epoch": epoch,
                    "step":  global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimiser.state_dict(),
                    "bs": bs
                }, ckpt)
                print(f"✅  New best model saved → {ckpt}\n")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    last_ckpt = f"{CHECK_DIR}/last_epoch_check{epoch}_step{global_step}_{ts}.pt"
    torch.save({
        "epoch": epoch,
        "step":  global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimiser.state_dict(),
        "bs": best_bs,
    }, last_ckpt)
    print(f"💾  Last model of epoch {epoch} saved → {last_ckpt}\n")

print("🎉 Training complete.")