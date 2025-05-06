# Required libraries
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import KFold

# Device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Dataset class
class PersonaStoryGraphDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.bert.eval()
        self.project = nn.Linear(768, 128).to(device)

        with open(jsonl_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                label = 1 if item['label'] == 1 else 0
                self.data.append((item['persona'], item['story'], label))

    def encode_sentences(self, sentences):
        with torch.no_grad():
            inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
            outputs = self.bert(**inputs)
            cls_embeddings = outputs.last_hidden_state[:,0,:]
            cls_embeddings = self.project(cls_embeddings)
        return cls_embeddings

    def build_graph(self, sentences):
        x = self.encode_sentences(sentences)
        num_nodes = x.size(0)
        if num_nodes == 1:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = torch.combinations(torch.arange(num_nodes, device=device), r=2).t()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        return Data(x=x, edge_index=edge_index)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        persona_sents, story_sents, label = self.data[idx]
        persona_graph = self.build_graph(persona_sents)
        story_graph = self.build_graph(story_sents)
        return persona_graph, story_graph, torch.tensor(label, dtype=torch.float32, device=device)
    
# (Pdb) fc_story_graph
# Data(x=[5, 128], edge_index=[2, 20])
# (Pdb) fc_persona_graph
# Data(x=[5, 128], edge_index=[2, 20])

# 2. GAT-based encoder
class GATEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=64):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=1)
        self.gat2 = GATConv(hidden_dim, output_dim, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if edge_index.size(1) == 0:
            x = torch.mean(x, dim=0, keepdim=True)
        else:
            x = F.relu(self.gat1(x, edge_index))
            x = self.gat2(x, edge_index)
            x = torch.mean(x, dim=0, keepdim=True)
        return x.squeeze(0)

# 3. Consistency Classifier
class ConsistencyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn_persona = GATEncoder()
        self.gnn_story = GATEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, persona_graph, story_graph):
        h_p = self.gnn_persona(persona_graph)
        h_s = self.gnn_story(story_graph)
        
        h_concat = torch.cat([h_p, h_s], dim=-1)
        logits = self.classifier(h_concat)
        prob = torch.sigmoid(logits)
        return prob

# 4. Accuracy computation function
def compute_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for persona_graph, story_graph, label in dataloader:
            prob = model(persona_graph, story_graph)
            pred = (prob >= 0.5).float()
            correct += (pred.view(-1) == label).sum().item()
            total += 1
    acc = correct / total
    return acc

# 5. Training code for one fold
def train_one_fold(model, train_loader, val_loader, fold, num_epochs=20, lr=3e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for persona_graph, story_graph, label in train_loader:
            optimizer.zero_grad()
            prob = model(persona_graph, story_graph)
            prob = prob.view(-1)
            loss = loss_fn(prob, label.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        val_acc = compute_accuracy(model, val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc*100:.2f}%")

    # Save model after each fold
    save_path = f"/Users/tzhang/projects/Personalized_RAG/model/classifer/consistency_classifier_fold{fold+1}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model for Fold {fold+1} saved to {save_path}")

# 6. Consistency Score computation for retriever connection
def compute_consistency_scores(model, persona_graph, story_graphs):
    model.eval()
    scores = []
    with torch.no_grad():
        for story_graph in story_graphs:
            prob = model(persona_graph, story_graph)
            scores.append(prob.view(-1))
    scores = torch.cat(scores)
    return scores  # (num_candidates,)

# 7. Main execution
if __name__ == "__main__":
    full_dataset = PersonaStoryGraphDataset("/Users/tzhang/projects/Personalized_RAG/data/consistency_annotation/annotated_persona_story.jsonl")
    indices = list(range(len(full_dataset)))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n===== Fold {fold+1} =====")
        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

        model = ConsistencyClassifier().to(device)
        train_one_fold(model, train_loader, val_loader, fold)

    print("Training completed.")

    # Example for retriever usage
    # model = ConsistencyClassifier().to(device)
    # model.load_state_dict(torch.load("consistency_classifier_fold5.pt"))
    # model.eval()
    # persona_graph = ... # single persona graph
    # story_graphs = [...] # list of candidate story graphs
    # scores = compute_consistency_scores(model, persona_graph, story_graphs)
    # print(scores)