#!/usr/bin/env python
# test.py ― GraphRAG-PD inference script
# --------------------------------------
import json, os, gc, argparse, random, torch, numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from transformers import AutoTokenizer
from GraphRAG_train import EnhancedBERTScoreFeedbackRAG         # ← your model class

# ---------------------------------------------------------------------
# 1. CLI ----------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--test_path",   default="/scratch1/tzhang62/GraphRAG_PD/Personalized_RAG/data/LAPDOG_new_dataset/valid_data_250.jsonl")
parser.add_argument("--story_path",  default="/scratch1/tzhang62/GraphRAG_PD/Personalized_RAG/data/story.jsonl")
parser.add_argument("--ckpt",        default="/scratch1/tzhang62/GraphRAG_PD/Personalized_RAG/graphrag_model/best_epoch_a100_1_step900_0p8959_20250509_105111.pt")
parser.add_argument("--cls_path",    default="/scratch1/tzhang62/GraphRAG_PD/Personalized_RAG/model/classifer/consistency_classifier_fold5.pt")
parser.add_argument("--out_dir",     default="/scratch1/tzhang62/GraphRAG_PD/Personalized_RAG/results/GraphRAG_eval/")
parser.add_argument("--batch",       type=int, default=4)
parser.add_argument("--sample_k",    type=int, default=50, help="story graphs sampled each turn")
parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device = torch.device(args.device)
print(f"👉 Running inference on {device}")

# ---------------------------------------------------------------------
# 2. Helpers -----------------------------------------------------------
def load_jsonl(path, cap=20):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if cap and i >= cap: break
            data.append(json.loads(line))
    return data

def construct_graph(texts, sent_encoder, thr=0.7):
    emb = sent_encoder.encode(texts, convert_to_tensor=True)
    sim = cosine_similarity(emb.cpu().numpy())
    ei = [[], []]
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

def extract_context(txt, turns=6):
    if "context:" not in txt: return []
    ctx = txt.split("context:")[1]
    ctx = ctx.replace("Q:", "Q: ").replace("R:", "R: ")
    qparts = [seg for seg in ctx.split("Q: ") if seg.strip()]
    turns_out = []
    for seg in qparts:
        q, *rest = seg.split("R: ")
        turns_out.append(f"Q: {q.strip()}")
        if rest: turns_out.append(f"R: {rest[0].strip()}")
    return turns_out[-turns:]

# ---------------------------------------------------------------------
# 3. Load resources ----------------------------------------------------
print("🔄 Loading data …")
test_data   = load_jsonl(args.test_path)
story_data  = load_jsonl(args.story_path)

sent_model  = SentenceTransformer("Lajavaness/bilingual-embedding-small", device=device, trust_remote_code=True)
tokenizer   = AutoTokenizer.from_pretrained("t5-small")

# Pre-build story graphs **once** (CPU)
print("📐 Building story graphs …")
story_graphs, story_sentences = [], []
for st in tqdm(story_data):
    sents = [s.strip() + '.' for s in st['text'].split('. ') if s.strip()]
    story_sentences.append(sents)
    if len(sents) > 1:
        story_graphs.append(construct_graph(sents, sent_model))
print(f"✅ {len(story_graphs)} graphs ready.")

# ---------------------------------------------------------------------
# 4. Dataset -----------------------------------------------------------
class PersonaDialogueStoryDatasetInfer(Dataset):
    def __init__(self, data):
        self.data = data
        # Pre-compute persona graphs on CPU
        print("📐 Building persona graphs …")
        self.pgraphs, self.persona_txt = [], []
        for item in tqdm(data):
            pers = extract_persona(item['question'])
            self.persona_txt.append(pers)
            pers_graph = construct_graph(pers if len(pers)>1 else pers*2, sent_model)
            self.pgraphs.append(pers_graph)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        d  = self.data[idx]
        ctx = extract_context(d['question'])
        enc = tokenizer(' '.join(ctx), padding='max_length', truncation=True,
                        max_length=512, return_tensors='pt')
        return {
            "persona_graph":  self.pgraphs[idx],
            "persona_sents":  self.persona_txt[idx],
            "context":        ctx,
            "input_ids":      enc.input_ids.squeeze(),
            "attn_mask":      enc.attention_mask.squeeze(),
            "reference":      d['answers'][0]   # gold answer
        }

dataset   = PersonaDialogueStoryDatasetInfer(test_data)
loader    = DataLoader(dataset, batch_size=args.batch, shuffle=False, collate_fn=lambda x: x)

# ---------------------------------------------------------------------
# 5. Model -------------------------------------------------------------
ckpt = torch.load(args.ckpt, map_location=device)
model = EnhancedBERTScoreFeedbackRAG(
            hidden_dim=128, out_dim=64,
            classifier_path=args.cls_path).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"🌟 Checkpoint loaded: epoch {ckpt.get('epoch', '?')}")

# ---------------------------------------------------------------------
# 6. Inference loop ----------------------------------------------------
out_path = os.path.join(
    args.out_dir, f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
results, sample_k = [], args.sample_k
print("🚀 Running inference …")

with torch.no_grad():
    for batch in tqdm(loader):
        for ex in batch:
            # pick K stories (incl. a similarity-based positive)
            persona_emb = sent_model.encode(' '.join(ex['persona_sents']),
                                            convert_to_tensor=True).to(device)
            cand_idx = np.random.choice(len(story_graphs),
                                        min(sample_k, len(story_graphs)),
                                        replace=False)
            # crude similarity search to ensure one strong positive
            best, best_i = -1, 0
            for i in cand_idx:
                sim = torch.cosine_similarity(
                    persona_emb,
                    torch.mean(story_graphs[i].x, 0, keepdim=True).to(device)
                ).item()
                if sim > best: best, best_i = sim, i
            if best_i not in cand_idx: cand_idx[0] = best_i
            pos_in_sample = int(np.where(cand_idx == best_i)[0][0])
            samp_graphs   = [story_graphs[i].to(device) for i in cand_idx]
            samp_sents    = [story_sentences[i]           for i in cand_idx]
            tgt = tokenizer(ex['reference'],
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt').input_ids.squeeze().to(device)

            outs = model(
                persona_graph   = ex['persona_graph'].to(device),
                story_graphs    = samp_graphs,
                positive_idx    = pos_in_sample,
                input_ids       = ex['input_ids'].unsqueeze(0).to(device),
                attention_mask  = ex['attn_mask'].unsqueeze(0).to(device),
                labels          = tgt.unsqueeze(0),                 # no teacher forcing
                references      = [ex['reference']],
                persona_sentences   = ex['persona_sents'],
                story_sentences_list= samp_sents
            )

            gen_ids = model.generator.generate(
                input_ids=ex['input_ids'].unsqueeze(0).to(device),
                attention_mask=ex['attn_mask'].unsqueeze(0).to(device),
                encoder_outputs=outs['encoder_outputs'],
                max_length=100)
            #import pdb; pdb.set_trace()
            gen_txt = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            print(gen_txt)

            results.append({
                "persona":      ex['persona_sents'],
                "dialogue_ctx": ex['context'],
                "response_gen": gen_txt,
                "response_ref": ex['reference']
            })

            # housekeeping
            del samp_graphs, outs, gen_ids
            torch.cuda.empty_cache()

        # flush every ≈100 examples
        if len(results) >= 100:
            with open(out_path, "a", encoding="utf-8") as f:
                for r in results: f.write(json.dumps(r, ensure_ascii=False)+"\n")
            results.clear()

# final flush
with open(out_path, "a", encoding="utf-8") as f:
    for r in results: f.write(json.dumps(r, ensure_ascii=False)+"\n")

print(f"✅ Done! Results saved to {out_path}")
