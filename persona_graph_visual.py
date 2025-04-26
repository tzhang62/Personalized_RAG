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
sample_persona = []

with open(train_data_path, 'r') as f:
    for i in range(4):  # Adjusted to read only 4 personas
        line = f.readline()
        line = json.loads(line)
        input_string = line['question']
        persona_start = input_string.find("persona:") + len("persona:")
        context_start = input_string.find("context:")
        persona = input_string[persona_start:context_start].strip()
        persona_list = persona.split(". ")
        persona_list[-1] = persona_list[-1].rstrip('.')
        print(persona_list)
        sample_persona.append(persona_list)

model = SentenceTransformer('Lajavaness/bilingual-embedding-small', trust_remote_code=True)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Create a subplot figure with 2x2 grid
fig = make_subplots(rows=2, cols=2, subplot_titles=[f"Persona {i+1}" for i in range(4)])

for idx, persona in enumerate(sample_persona):
    # Encode persona attributes
    persona_embeddings = model.encode(persona, convert_to_tensor=True).to(device)
    print(persona_embeddings.shape)
    #import pdb; pdb.set_trace()

    # Compute similarity on CPU
    embeddings_cpu = persona_embeddings.cpu().numpy()
    similarity_matrix = cosine_similarity(embeddings_cpu)
    print(f"Similarity Matrix for Persona {idx+1}:\n", similarity_matrix)

    # Construct edges based on similarity threshold
    threshold = 0.7
    edge_index = [[], []]
    num_nodes = persona_embeddings.size(0)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if similarity_matrix[i][j] >= threshold:
                edge_index[0].extend([i, j])
                edge_index[1].extend([j, i])

    # Create a subgraph for the current persona
    graph_data = Data(x=persona_embeddings.cpu(), edge_index=torch.tensor(edge_index, dtype=torch.long).cpu())
    subgraph_nx = to_networkx(graph_data, to_undirected=True)

    # Create a layout for the subgraph
    pos = nx.spring_layout(subgraph_nx, seed=42, k=0.4)

   

    # Create node traces

     # Create node traces
    min_vertical_distance = 0.12
    max_iterations = 50
    for iteration in range(max_iterations):
        adjusted = False
        for node1, node2 in itertools.combinations(subgraph_nx.nodes(), 2):
            x1, y1 = pos[node1]
            x2, y2 = pos[node2]
            if abs(y1 - y2) < min_vertical_distance:
                adjusted = True
                shift = min_vertical_distance/2
                # Push them apart vertically in opposite directions
                if y1 <= y2:
                    pos[node1] = (x1, y1 - shift)
                    pos[node2] = (x2, y2 + shift)
                else:
                    pos[node1] = (x1, y1 + shift)
                    pos[node2] = (x2, y2 - shift)
        if not adjusted:
            break
    node_x = []
    node_y = []
    node_text = []
    text_positions = []
    for node in subgraph_nx.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        # Wrap text for long captions
        words = persona[node].split()
        wrapped_text = '<br>'.join([' '.join(words[i:i+10]) for i in range(0, len(words), 10)])
        node_text.append(wrapped_text)
        # Determine text position based on x-coordinate
        if x < 0 and y > 0:
            t_pos = "middle right"
        if x < 0 and y <= 0:
            t_pos = "middle right"
        if x >= 0 and y > 0:
            t_pos = "middle left"
        if x >= 0 and y <= 0:
            t_pos = "middle left"
        text_positions.append(t_pos)

    edge_x = []
    edge_y = []
    for edge in subgraph_nx.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='gray'),
        hoverinfo='none',
        mode='lines')


    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            size=20,
            color='lightblue',
            line_width=1),
        text=node_text,
        textposition=text_positions,
        textfont=dict(size=18, color='black', family='Arial')
    )

    # Determine the subplot position
    row = idx // 2 + 1
    col = idx % 2 + 1

    # Add traces to the subplot
    fig.add_trace(edge_trace, row=row, col=col)
    fig.add_trace(node_trace, row=row, col=col)

# Update layout
fig.update_layout(
    showlegend=False,
    width=1200,
    height=1200
)

# Hide grid lines and axes
fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

fig.show()