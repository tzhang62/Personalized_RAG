import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import json

train_data_path = '/Users/tzhang/projects/LAPDOG/data/LAPDOG_new_dataset/train_data_10000.jsonl'
sample_persona = []
with open(train_data_path, 'r') as f:
    for i in range(16):
        line = f.readline()
        line = json.loads(line)
        input_string = line['question']
        persona_start = input_string.find("persona:") + len("persona:")
        context_start = input_string.find("context:")
        persona = input_string[persona_start:context_start].strip()
        persona_list = persona.split(". ")
        persona_list[-1] = persona_list[-1].rstrip('.')
        sample_persona.append(persona_list)
model = SentenceTransformer('all-MiniLM-L6-v2')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load or define your persona data
# sample_persona = [...]  # Your list of 16 personas, each a list of attribute strings

# Flatten all persona attributes
all_attributes = []
persona_index = []

for i, persona in enumerate(sample_persona):
    all_attributes.extend(persona)
    persona_index.extend([f'Persona {i+1}'] * len(persona))

# Encode all attributes
all_embeddings = model.encode(all_attributes, convert_to_tensor=True).cpu().numpy()

# Reduce to 2D with PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(all_embeddings)

# Compute cosine similarity
sim_matrix = cosine_similarity(all_embeddings)
threshold = 0.5
edge_x, edge_y = [], []

for i in range(len(all_embeddings)):
    for j in range(i + 1, len(all_embeddings)):
        if sim_matrix[i][j] > threshold:
            x0, y0 = reduced_embeddings[i]
            x1, y1 = reduced_embeddings[j]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

# Edges
edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1, color='#888'),
    hoverinfo='none',
    mode='lines')

# Assign colors to personas
colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'gold',
          'teal', 'pink', 'brown', 'gray', 'olive', 'navy', 'coral', 'indigo']
color_map = {f'Persona {i+1}': colors[i % len(colors)] for i in range(16)}
node_colors = [color_map[p] for p in persona_index]

# Nodes
node_trace = go.Scatter(
    x=reduced_embeddings[:, 0],
    y=reduced_embeddings[:, 1],
    mode='markers+text',
    text=all_attributes,
    hoverinfo='text',
    marker=dict(size=10, color=node_colors, line=dict(width=1, color='black')),
    textposition="top center",
    textfont=dict(size=9)
)

# Final figure
fig = go.Figure(data=[edge_trace, node_trace])
fig.update_layout(
    title="Unified Persona Attribute Graph (PCA Projection)",
    showlegend=False,
    width=1000,
    height=800,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor='white'
)

fig.show()
