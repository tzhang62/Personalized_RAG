# Add to KG_construction.py

import torch.nn.functional as F
from torch.optim import Adam
from sentence_transformers import SentenceTransformer
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

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

class DualGraphEncoder(torch.nn.Module):
    def __init__(self, in_dim=384, hid_dim=128, out_dim=64):
        super(DualGraphEncoder, self).__init__()
        # Persona encoder
        self.persona_gnn = PersonaGNN(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim)
        # Story encoder (identical architecture but separate weights)
        self.story_gnn = PersonaGNN(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim) 
        # Temperature parameter (learnable)
        self.temperature = torch.nn.Parameter(torch.ones([]) * 0.07)
        
    def encode_persona(self, persona_graph):
        """Encode a persona graph to a vector"""
        return self.persona_gnn(persona_graph.x, persona_graph.edge_index)
        
    def encode_story(self, story_graph):
        """Encode a story graph to a vector"""
        return self.story_gnn(story_graph.x, story_graph.edge_index)
    
    def forward(self, persona_graph, story_graphs):
        """
        Compute similarity scores between a persona and multiple stories
        
        Args:
            persona_graph: PyG Data object for the persona graph
            story_graphs: List of PyG Data objects for story graphs
            
        Returns:
            similarity_scores: Tensor of similarity scores
        """
        # Encode persona (already outputs a mean-pooled graph representation)
        persona_emb = self.encode_persona(persona_graph)
        
        # Encode all stories
        story_embs = []
        for story_graph in story_graphs:
            story_emb = self.encode_story(story_graph)
            story_embs.append(story_emb)
        story_embs = torch.stack(story_embs)
        
        # Compute similarity scores (scaled dot product)
        similarity_scores = torch.matmul(
            F.normalize(persona_emb, dim=0).unsqueeze(0),
            F.normalize(story_embs, dim=1).t()
        ) / self.temperature
        
        return similarity_scores.squeeze()
    
    def compute_contrastive_loss(self, similarity_scores, positive_idx):
        """
        Compute contrastive loss with in-batch negatives
        
        Args:
            similarity_scores: Tensor of similarity scores
            positive_idx: Index of the positive example in the batch
            
        Returns:
            loss: Contrastive loss value
        """
        return F.cross_entropy(similarity_scores.unsqueeze(0), 
                              torch.tensor([positive_idx], device=similarity_scores.device))


# Example of how to use the retriever
def train_retriever(personas, stories, positive_pairs, num_epochs=10, lr=0.001):
    """
    Train the dual graph encoder retriever
    
    Args:
        personas: List of persona graphs (PyG Data objects)
        stories: List of story graphs (PyG Data objects)
        positive_pairs: List of (persona_idx, story_idx) tuples indicating positive pairs
        num_epochs: Number of training epochs
        lr: Learning rate
    """
    # Initialize the retriever
    retriever = DualGraphEncoder().to(device)
    optimizer = Adam(retriever.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for persona_idx, pos_story_idx in positive_pairs:
            # Get persona graph
            persona_graph = personas[persona_idx].to(device)
            
            # Get a batch of story graphs (including the positive)
            batch_size = min(16, len(stories))  # Limit batch size
            story_indices = [pos_story_idx] + [idx for idx in range(len(stories)) 
                                            if idx != pos_story_idx][:batch_size-1]
            batch_stories = [stories[idx].to(device) for idx in story_indices]
            
            # Forward pass
            similarity_scores = retriever(persona_graph, batch_stories)
            
            # Compute loss (positive story is at index 0)
            loss = retriever.compute_contrastive_loss(similarity_scores, 0)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
    
    return retriever


# Function to retrieve stories using the trained retriever
def retrieve_stories(retriever, persona_graph, story_graphs, top_k=3):
    """
    Retrieve top-k stories for a persona
    
    Args:
        retriever: Trained DualGraphEncoder
        persona_graph: PyG Data object for the persona graph
        story_graphs: List of PyG Data objects for story graphs
        top_k: Number of stories to retrieve
        
    Returns:
        top_indices: Indices of top-k stories
        scores: Similarity scores for top-k stories
    """
    # Set model to evaluation mode
    retriever.eval()
    
    with torch.no_grad():
        # Compute similarity scores
        similarity_scores = retriever(persona_graph, story_graphs)
        
        # Get top-k indices and scores
        top_scores, top_indices = torch.topk(similarity_scores, min(top_k, len(story_graphs)))
        
    return top_indices.cpu().numpy(), top_scores.cpu().numpy()


class RetrievalAugmentedGenerator(torch.nn.Module):
    def __init__(self, retriever, generator, lambda_coef=0.5):
        super().__init__()
        self.retriever = retriever
        self.generator = generator  # Your T5 or other generator model
        self.lambda_coef = lambda_coef
        
    def forward(self, persona_graph, story_graphs, input_ids, 
                attention_mask, labels=None, positive_story_idx=None):
        """
        Forward pass with joint retrieval and generation
        
        Args:
            persona_graph: Persona graph
            story_graphs: List of story graphs
            input_ids, attention_mask: Inputs for the generator
            labels: Target outputs for the generator (optional)
            positive_story_idx: Index of the positive story (for training)
            
        Returns:
            Dictionary with loss and logits
        """
        # Get similarity scores from retriever
        similarity_scores = self.retriever(persona_graph, story_graphs)
        
        # Get top story index
        top_story_idx = torch.argmax(similarity_scores).item()
        top_story = story_graphs[top_story_idx]
        
        # Generate response
        # Here you need to convert the story graph back to text or embeddings
        # that your generator can use
        story_content = convert_graph_to_text(top_story)  # Implement this function
        
        # Append story content to input
        augmented_input = augment_input_with_story(input_ids, story_content)  # Implement this
        
        # Generate response
        outputs = self.generator(
            input_ids=augmented_input,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # If in training mode and we have a positive story
        if self.training and positive_story_idx is not None:
            # Compute retriever loss
            retrieval_loss = self.retriever.compute_contrastive_loss(
                similarity_scores, positive_story_idx)
            
            # Combine losses
            if labels is not None:
                total_loss = outputs.loss + self.lambda_coef * retrieval_loss
                outputs.loss = total_loss
        
        return outputs