import networkx as nx
import plotly.graph_objects as go
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from plotly.subplots import make_subplots
import torch
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import itertools

train_data_path = '/Users/tzhang/projects/LAPDOG/data/LAPDOG_new_dataset/train_data_10000.jsonl'

all_utterances = []
with open(train_data_path, 'r') as f:
    for i in range(4):  # Adjusted to read only 4 personas
        line = f.readline()
        line = json.loads(line)
        input_string = line['question']
        persona_start = input_string.find("persona:") + len("persona:")
        context_start = input_string.find("context:")
        utterances = []
        current_utterance = ""
        persona = input_string[persona_start:context_start].strip()
        context = input_string[context_start:].strip('context: ')
        for utterance in context.split("Q:"):
            if utterance.strip():
                if 'R:' in utterance:
                    q_part, r_part = utterance.split("R:",1)
                    if q_part.strip():
                        utterances.append(q_part.strip())
                    if r_part.strip():
                        utterances.append(r_part.strip())
                else:
                    utterances.append(utterance.strip())
        all_utterances.append(utterances)
        
dialogue = all_utterances[0]
import pdb; pdb.set_trace()
# === 2. Encode ===
model = SentenceTransformer('Lajavaness/bilingual-embedding-small', trust_remote_code=True)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
embeddings = model.encode(dialogue, convert_to_tensor=True).to(device)
embeddings_cpu = embeddings.cpu().numpy()

# === 3. Edges by similarity ===
threshold = 0.7
edge_index = [[], []]
num_nodes = len(dialogue)
similarity_matrix = cosine_similarity(embeddings_cpu)

for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        if similarity_matrix[i][j] >= threshold:
            edge_index[0].extend([i, j])
            edge_index[1].extend([j, i])

# === 4. Build graph ===
G = nx.Graph()
for i in range(num_nodes):
    G.add_node(i)
for src, tgt in zip(edge_index[0], edge_index[1]):
    G.add_edge(src, tgt)

pos = nx.spring_layout(G, seed=42, k=0.4)

# Resolve vertical overlaps for nodes
min_vertical_distance = 0.12
max_iterations = 50
for iteration in range(max_iterations):
    adjusted = False
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            xi, yi = pos[i]
            xj, yj = pos[j]
            # Only check vertical distance if nodes are on the same side
            if (xi > 0 and xj > 0) or (xi <= 0 and xj <= 0):
                if abs(yi - yj) < min_vertical_distance:
                    adjusted = True
                    shift = min_vertical_distance / 2
                    if yi <= yj:
                        pos[i] = (xi, yi - shift)
                        pos[j] = (xj, yj + shift)
                    else:
                        pos[i] = (xi, yi + shift)
                        pos[j] = (xj, yj - shift)
    if not adjusted:
        break

connected_nodes = set()
for edge in G.edges():
    connected_nodes.update(edge)

node_colors = []
for i in range(num_nodes):
    is_connected = i in connected_nodes
    is_q = (i % 2 == 0)
    if is_q:
        color = "#1E90FF" if is_connected else "#87CEEB"  # Bright blue for questions
    else:
        color = "#FF6347" if is_connected else "#FFA07A"  # Bright red for responses
    node_colors.append(color)

node_x, node_y = [], []
for i in range(num_nodes):
    x, y = pos[i]
    node_x.append(x)
    node_y.append(y)

# === 5. Create edge trace ===
edge_x, edge_y = [], []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

# === 7. Add text annotations for utterances ===
annotations = []
min_vertical_distance = 0.25  # Minimum vertical distance between annotations
max_iterations = 100  # Maximum number of iterations for overlap resolution

# Resolve vertical overlaps for both nodes and annotations
for iteration in range(max_iterations):
    adjusted = False
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if i != j:  # Skip comparing the same node
                xi, yi = pos[i]
                xj, yj = pos[j]
                
                # Only check vertical distance if nodes are on the same side
                if (xi > 0 and xj > 0) or (xi <= 0 and xj <= 0):
                    if abs(yi - yj) < min_vertical_distance:
                        adjusted = True
                        shift = min_vertical_distance / 2
                        if yi <= yj:
                            pos[i] = (xi, yi - shift)
                            pos[j] = (xj, yj + shift)
                        else:
                            pos[i] = (xi, yi + shift)
                            pos[j] = (xj, yj - shift)
    
    if not adjusted:
        break

# Update node positions after adjustment
node_x = []
node_y = []
for i in range(num_nodes):
    x, y = pos[i]
    node_x.append(x)
    node_y.append(y)

# Update edge positions
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

# Recreate node trace with updated positions
node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers+text',
    hoverinfo='text',
    text=[str(i+1) for i in range(num_nodes)],  # Add node numbers starting from 1
    textposition='middle center',
    textfont=dict(size=30, color='white', family="Arial Black"),  # Larger, bolder text
    marker=dict(
        size=60,  # Larger nodes
        color=node_colors,
        line=dict(width=2, color='black')  # Thicker border
    )
)

# Create annotation positions based on node positions
annotation_positions = []
for i in range(num_nodes):
    x, y = pos[i]
    if x > 0:  # Right side
        x_offset = -0.19
        x_anchor = 'right'
    else:  # Left side
        x_offset = 0.20
        x_anchor = 'left'
    if y > 0 and x > 0:
        y_offset = -0.12
    if y > 0 and x < 0:
        y_offset = 0.08
    annotation_positions.append((x + x_offset, y + y_offset, i, x_anchor))

# Create annotations with node positions
for x, y, node_idx, x_anchor in annotation_positions:
    words = dialogue[node_idx].split()
    lines = [' '.join(words[j:j+5]) for j in range(0, len(words), 5)]
    wrapped_text = "<br>".join(lines)
    
    speaker_icon = "🧑" if node_idx % 2 == 0 else "🤖"
    annotations.append(dict(
        x=x,
        y=y,
        xref="x",
        yref="y",
        text=f"{speaker_icon}{':'}{wrapped_text}",
        showarrow=False,
        font=dict(size=23, color='black'),
        align=x_anchor
    ))

# === 8. Plot ===
fig = go.Figure()

# Edges
fig.add_trace(go.Scatter(
    x=edge_x, y=edge_y,
    mode='lines',
    line=dict(width=1.5, color='gray'),
    hoverinfo='none'
))

# Nodes with numbers
fig.add_trace(node_trace)

fig.update_layout(
    title="",
    showlegend=False,
    width=1800,  # Increased width to accommodate larger text
    height=1400,  # Increased height for better spacing
    paper_bgcolor='white',
    plot_bgcolor='white',
    margin=dict(l=20, r=20, t=40, b=20),
    annotations=annotations
)

fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

fig.show()