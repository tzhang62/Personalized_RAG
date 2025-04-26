# A Knowledge Graph (KG) consists of:
# Nodes: Represent persona attributes or relevant entities.
# Edges: Represent semantic or relational connections between these entities.
# Edge Types (optional but beneficial): Clearly defined relations like similar-to, related-to, cause-of, etc.

#step1: extract persona attributes from the persona description
#step2: define the node embeddings for each attribute

from sentence_transformers import SentenceTransformer
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch.nn.functional as F

persona = ["I like to remodel homes.", "I like to go hunting.", "I like to shoot a bow.", "my favorite holiday is Halloween."]

model = SentenceTransformer('Lajavaness/bilingual-embedding-small', trust_remote_code=True)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

persona_embeddings = model.encode(persona, convert_to_tensor=True)



# Ensure your persona_embeddings tensor is already computed (assumed)
persona_embeddings = persona_embeddings.to(device)

# Compute similarity on CPU (required by sklearn), then move tensors back
embeddings_cpu = persona_embeddings.cpu().numpy()
similarity_matrix = cosine_similarity(embeddings_cpu)

# Construct edges based on similarity threshold
threshold = 0.7
edge_index = [[], []]
num_nodes = persona_embeddings.size(0)

for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        if similarity_matrix[i][j] >= threshold:
            edge_index[0].extend([i, j])
            edge_index[1].extend([j, i])

# Move edge_index to MPS device explicitly
edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)

# Define the GNN model explicitly for MPS
class PersonaGNN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(PersonaGNN, self).__init__()
        self.gat1 = GATConv(in_dim, hid_dim).to(device)
        self.gat2 = GATConv(hid_dim, out_dim).to(device)
            
    def forward(self, x, edge_index):
        x = torch.relu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x.mean(dim=0)

# Prepare PyG data object explicitly on MPS
graph_data = Data(x=persona_embeddings, edge_index=edge_index).to(device)

# Initialize the model on MPS device
persona_gnn = PersonaGNN(in_dim=persona_embeddings.size(1), hid_dim=128, out_dim=64).to(device)

# Forward pass explicitly on MPS device
persona_embedding_final = persona_gnn(graph_data.x, graph_data.edge_index)

# Print results
print(graph_data)
print(persona_embedding_final)

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

# Convert PyG graph to NetworkX
graph_nx = to_networkx(graph_data, to_undirected=True)
labels = {i: attr for i, attr in enumerate(persona)}
plt.figure(figsize=(4,4))

# Draw nodes and edges
pos = nx.spring_layout(graph_nx, seed=42)  # Layout for clarity
nx.draw_networkx_nodes(graph_nx, pos, node_color='skyblue', node_size=1500)
nx.draw_networkx_edges(graph_nx, pos, edge_color='gray')

# Draw labels (persona attributes clearly shown)
nx.draw_networkx_labels(
    graph_nx,
    pos,
    labels=labels,
    font_size=10,
    font_weight='bold',
    font_color='black'
)

plt.title("Persona Graph with Explicit Node Meanings", fontsize=15)
plt.axis('off')
plt.show()

class DualGraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Persona encoder
        self.persona_gnn = PersonaGNN(input_dim, hidden_dim, output_dim)
        # Story encoder
        self.story_gnn = PersonaGNN(input_dim, hidden_dim, output_dim)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
    def forward(self, persona_graph, story_graphs):
        # Encode persona
        persona_emb = self.persona_gnn(persona_graph.x, persona_graph.edge_index)
        
        # Encode stories
        story_embs = []
        for story_graph in story_graphs:
            story_emb = self.story_gnn(story_graph.x, story_graph.edge_index)
            story_embs.append(story_emb)
        story_embs = torch.stack(story_embs)
        
        # Compute similarities
        similarities = torch.matmul(persona_emb, story_embs.t()) / self.temperature
        
        return similarities
        
    def compute_loss(self, similarities, positive_idx):
        # Contrastive loss with positive story index
        return F.cross_entropy(similarities, torch.tensor([positive_idx]))

class RetrievalAugmentedGenerator(nn.Module):
    def __init__(self, retriever, generator):
        super().__init__()
        self.retriever = retriever
        self.generator = generator
        self.lambda_coef = 0.5
        
    def forward(self, persona_graph, story_graphs, target_response):
        # Retrieval step
        similarities = self.retriever(persona_graph, story_graphs)
        retrieved_stories = self.get_top_k_stories(similarities)
        
        # Generation step
        generation_output = self.generator(retrieved_stories, target_response)
        
        # Compute losses
        retrieval_loss = self.retriever.compute_loss(similarities, positive_idx)
        generation_loss = generation_output.loss
        
        total_loss = retrieval_loss + self.lambda_coef * generation_loss
        
        return total_loss, generation_output.logits
