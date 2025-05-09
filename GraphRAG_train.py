# Add to KG_construction.py

import torch.nn.functional as F
from torch.optim import Adam
from sentence_transformers import SentenceTransformer
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
from transformers import GPT2LMHeadModel, T5ForConditionalGeneration, AutoModel, AutoTokenizer
from Classifier_Retriever import ConsistencyClassifier, PersonaStoryGraphDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
### graph construction

persona = ["I like to remodel homes.", "I like to go hunting.", "I like to shoot a bow.", "my favorite holiday is Halloween."]

model = SentenceTransformer('Lajavaness/bilingual-embedding-small', trust_remote_code=True)


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

graph_data = Data(x=persona_embeddings, edge_index=edge_index).to(device)

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


class GraphToTextGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Graph encoder
        self.graph_encoder = nn.ModuleList([
            GATConv(384, 256),
            GATConv(256, 768)
        ])
        # Text decoder (e.g., GPT-2)
        self.text_decoder = GPT2LMHeadModel.from_pretrained("gpt2")
        # Cross-attention to connect graph encodings to text decoder
        self.cross_attention = nn.MultiheadAttention(768, 8)
        # Projection layer to map cross-attended representations back to model dimension
        self.projection = nn.Linear(768, 768)
        
    def forward(self, graph, decoder_input_ids, labels=None, generate=False, max_length=100):
        """
        Forward pass for graph-to-text generation
        
        Args:
            graph: PyG Data object containing the graph
            decoder_input_ids: Input token IDs for the text decoder
            labels: Optional target labels for training
            generate: Whether to run generation (inference) or training
            max_length: Maximum generation length for inference
            
        Returns:
            During training: loss, logits
            During inference: generated text IDs
        """
        # Process graph with GAT layers
        x, edge_index = graph.x, graph.edge_index
        for i, gat in enumerate(self.graph_encoder):
            x = gat(x, edge_index)
            if i < len(self.graph_encoder) - 1:
                x = F.relu(x)
        
        # Graph node representations after GAT processing
        graph_node_embeddings = x  # Shape: [num_nodes, 768]
        
        if generate:
            # For inference/generation mode
            return self.generate_text(graph_node_embeddings, decoder_input_ids, max_length)
        else:
            # For training mode
            # Get decoder outputs normally first
            decoder_outputs = self.text_decoder(
                input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            decoder_hidden_states = decoder_outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            
            batch_size, seq_len, hidden_size = decoder_hidden_states.size()
            
            # Prepare graph embeddings for cross-attention (expand to match batch size)
            # Assuming we process one graph per batch item
            expanded_graph_embeddings = graph_node_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Apply cross-attention between decoder states and graph nodes
            # Reshape for cross-attention: [seq_len, batch_size, hidden_size]
            q = decoder_hidden_states.transpose(0, 1)
            k = expanded_graph_embeddings.transpose(0, 1)
            v = expanded_graph_embeddings.transpose(0, 1)
            
            enhanced_states, _ = self.cross_attention(q, k, v)
            
            # Reshape back: [batch_size, seq_len, hidden_size]
            enhanced_states = enhanced_states.transpose(0, 1)
            
            # Project back to model dimension if needed
            enhanced_states = self.projection(enhanced_states)
            
            # Add residual connection
            enhanced_states = enhanced_states + decoder_hidden_states
            
            # Pass through the LM head to get logits
            lm_head = self.text_decoder.get_output_embeddings()
            logits = lm_head(enhanced_states)
            
            # Compute loss if labels are provided
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                # Reshape logits and labels for loss computation
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            return {"loss": loss, "logits": logits}
    
    def generate_text(self, graph_node_embeddings, input_ids, max_length):
        """
        Autoregressive text generation using graph information
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Start with input_ids
        curr_ids = input_ids
        
        for _ in range(max_length):
            # Get model outputs
            outputs = self.text_decoder(curr_ids, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            
            # Only need to enhance the last position for generation
            last_hidden = hidden_states[:, -1:, :]
            
            # Expand graph node embeddings to batch size
            expanded_graph_embeddings = graph_node_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Apply cross-attention for the last position
            q = last_hidden.transpose(0, 1)  # [1, batch_size, hidden_size]
            k = expanded_graph_embeddings.transpose(0, 1)  # [num_nodes, batch_size, hidden_size]
            v = expanded_graph_embeddings.transpose(0, 1)
            
            enhanced_last_hidden, _ = self.cross_attention(q, k, v)
            enhanced_last_hidden = enhanced_last_hidden.transpose(0, 1)  # [batch_size, 1, hidden_size]
            
            # Project and add residual
            enhanced_last_hidden = self.projection(enhanced_last_hidden) + last_hidden
            
            # Get logits for next token prediction
            lm_head = self.text_decoder.get_output_embeddings()
            next_token_logits = lm_head(enhanced_last_hidden).squeeze(1)  # [batch_size, vocab_size]
            
            # Sample next token
            next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)  # [batch_size, 1]
            
            # Concatenate with previous tokens
            curr_ids = torch.cat([curr_ids, next_tokens], dim=-1)
            
            # Check if we've generated EOS token
            if (next_tokens == self.text_decoder.config.eos_token_id).all():
                break
                
        return curr_ids

class BERTScoreFeedbackRAG(nn.Module):
    def __init__(self, hidden_dim=128, out_dim=64):
        super().__init__()
        # Retriever component
        self.retriever = DualGraphEncoder(in_dim=384, hid_dim=hidden_dim, out_dim=out_dim)
        
        # Generator component
        self.generator = T5ForConditionalGeneration.from_pretrained("t5-small")
        
        # Connection layer from graph to text model
        self.graph_to_text = nn.Linear(out_dim, self.generator.config.d_model)
        
        # BERTScore model for feedback
        # We'll use bert-base-uncased for efficiency, but you can use larger models too
        self.bert_model = AutoModel.from_pretrained("bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Hyperparameters
        self.lambda_retrieval = 0.3  # Weight for retrieval loss
        # self.lambda_bertscore = 0.2  # Weight for BERTScore feedback
        self.lambda_bertscore = 0.8
    def forward(self, persona_graph, story_graphs, positive_idx=None, 
                input_ids=None, attention_mask=None, labels=None,
                references=None):
        """
        Joint forward pass with BERTScore feedback
        
        Args:
            persona_graph: The persona graph
            story_graphs: List of candidate story graphs
            positive_idx: Index of the positive story (for training)
            input_ids, attention_mask, labels: Inputs for text generation
            references: Reference responses for BERTScore computation
            
        Returns:
            Dictionary with combined loss and generation outputs
        """
        # 1. Retrieval step
        similarity_scores = self.retriever(persona_graph, story_graphs)
        
        # Get probabilities over stories
        story_probs = F.softmax(similarity_scores, dim=0)
        
        # During training, sample stories based on similarity scores
        if self.training:
            # Use Gumbel-softmax for differentiable sampling
            story_sample = F.gumbel_softmax(similarity_scores, tau=1.0, hard=True)
            selected_idx = torch.argmax(story_sample).item()
        else:
            # During inference, use the highest scoring story
            selected_idx = torch.argmax(similarity_scores).item()
            
        selected_story = story_graphs[selected_idx]
        
        # 2. Encode selected story graph for the generator
        story_embedding = self.retriever.encode_story(selected_story)
        text_embedding = self.graph_to_text(story_embedding)
        
        # 3. Generate with the graph-enhanced input
        encoder_outputs = self.prepare_encoder_with_graph(input_ids, text_embedding)
        
        outputs = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            encoder_outputs=encoder_outputs,
            output_hidden_states=True
        )
        
        # 4. Compute standard generation loss
        generation_loss = outputs.loss
        
        # Combined loss starts with generation loss
        total_loss = generation_loss
        
        # 5. Add retrieval loss if in training
        retrieval_loss = None
        if positive_idx is not None:
            retrieval_loss = self.retriever.compute_contrastive_loss(
                similarity_scores, positive_idx)
            total_loss += self.lambda_retrieval * retrieval_loss
        
        # 6. Add BERTScore feedback if references provided
        bertscore_loss = None
        if references is not None:
            # Generate text (without teacher forcing)
            with torch.no_grad():
                generated_ids = self.generator.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    encoder_outputs=encoder_outputs,
                    max_length=100
                )
                
                # Decode generated texts
                generated_texts = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True)
                
            # Calculate BERTScore
            bertscore_loss = self.compute_bertscore_loss(generated_texts, references)
            
            # Add to total loss
            total_loss += self.lambda_bertscore * bertscore_loss
        
        # Return a dictionary with all components
        return {
            'loss': total_loss,
            'generation_loss': generation_loss,
            'retrieval_loss': retrieval_loss,
            'bertscore_loss': bertscore_loss,
            'logits': outputs.logits
        }
    
    def prepare_encoder_with_graph(self, input_ids, graph_embedding):
        """Prepare T5 encoder outputs enhanced with graph embedding"""
        # Get original encoder outputs
        encoder_outputs = self.generator.encoder(input_ids)
        original_hidden_states = encoder_outputs.last_hidden_state
        
        # Fuse graph embedding with all encoder tokens
        batch_size = input_ids.shape[0]
        graph_emb_expanded = graph_embedding.unsqueeze(0).unsqueeze(1).expand(batch_size, original_hidden_states.size(1), -1)
        
        # Add graph information to original hidden states (weighted addition)
        alpha = 0.3  # Weight for graph information
        enhanced_hidden_states = (1-alpha) * original_hidden_states + alpha * graph_emb_expanded
        
        # Update encoder outputs without changing sequence length
        encoder_outputs.last_hidden_state = enhanced_hidden_states
        return encoder_outputs
    
    def compute_bertscore_loss(self, candidates, references):
        """
        Compute BERTScore-based loss to encourage coherence
        
        Args:
            candidates: List of generated texts
            references: List of reference texts
            
        Returns:
            bertscore_loss: Loss based on BERTScore (1 - F1 score)
        """
        # Tokenize candidates and references
        candidate_encodings = self.tokenizer(candidates, return_tensors="pt", 
                                           padding=True, truncation=True).to(self.bert_model.device)
        reference_encodings = self.tokenizer(references, return_tensors="pt", 
                                           padding=True, truncation=True).to(self.bert_model.device)
        
        # Get BERT embeddings
        with torch.no_grad():
            candidate_outputs = self.bert_model(**candidate_encodings)
            reference_outputs = self.bert_model(**reference_encodings)
            
            # Get token embeddings from last hidden state
            candidate_embeds = candidate_outputs.last_hidden_state
            reference_embeds = reference_outputs.last_hidden_state
            
            # Normalize embeddings
            candidate_embeds = F.normalize(candidate_embeds, p=2, dim=2)
            reference_embeds = F.normalize(reference_embeds, p=2, dim=2)
        
        # Compute cosine similarity matrix for each pair
        batch_scores = []
        for i in range(len(candidates)):
            # Get valid token embeddings (exclude padding)
            cand_mask = candidate_encodings.attention_mask[i].bool()
            ref_mask = reference_encodings.attention_mask[i].bool()
            
            c_embed = candidate_embeds[i, cand_mask]
            r_embed = reference_embeds[i, ref_mask]
            
            # Compute similarity matrix
            sim_matrix = torch.matmul(c_embed, r_embed.transpose(0, 1))
            
            # Compute precision, recall and F1
            precision = sim_matrix.max(dim=1)[0].mean()
            recall = sim_matrix.max(dim=0)[0].mean()
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            batch_scores.append(f1)
        
        # Average BERTScore F1 across batch
        avg_bertscore_f1 = torch.stack(batch_scores).mean()
        
        # Loss is 1 - F1 score (so higher F1 means lower loss)
        bertscore_loss = 1 - avg_bertscore_f1
        
        return bertscore_loss

class EnhancedBERTScoreFeedbackRAG(nn.Module):
    def __init__(self, hidden_dim=128, out_dim=64, classifier_path=None):
        super().__init__()
        # Standard components on primary device (MPS/GPU)
        self.retriever = DualGraphEncoder(in_dim=384, hid_dim=hidden_dim, out_dim=out_dim)
        self.generator = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.graph_to_text = nn.Linear(out_dim, self.generator.config.d_model)
        self.bert_model = AutoModel.from_pretrained("bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Force CPU for consistency classifier
        self.consistency_classifier = ConsistencyClassifier().to('cuda')
        if classifier_path:
            # Load classifier to CPU explicitly
            self.consistency_classifier.load_state_dict(
                torch.load(classifier_path, map_location='cuda')
            )
            print(f"Loaded consistency classifier from {classifier_path}")
        
        # Freeze classifier by default
        for param in self.consistency_classifier.parameters():
            param.requires_grad = False
        
        # Initialize BERT for creating fully connected graphs - on CPU
        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model_for_graphs = AutoModel.from_pretrained('bert-base-uncased').to('cpu')
        
        with torch.no_grad():
            self.bert_project = nn.Linear(768, 128).to('cuda')
            
        # Story embedding expansion to fix dimension mismatch
        self.story_expansion = nn.Linear(64, 128).to('cuda')
            
        # Hyperparameters
        self.lambda_retrieval = 0.3
        # self.lambda_bertscore = 0.2
        self.lambda_bertscore = 0.7
        self.lambda_consistency = 0.2
        self.consistency_threshold = 0.5
    
    def encode_sentences_for_graph(self, sentences):
        """Encode sentences using BERT for consistency classifier - on CPU"""
        with torch.no_grad():
            inputs = self.bert_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to('cuda')
            outputs = self.bert_model_for_graphs(**inputs)
            cls_embeddings = outputs.last_hidden_state[:,0,:]
            cls_embeddings = self.bert_project(cls_embeddings)
        return cls_embeddings
        
    def build_fully_connected_graph(self, sentences):
        """Create a fully connected graph from sentences for consistency classifier - on CPU"""
        # Encode sentences on CPU
        x = self.encode_sentences_for_graph(sentences)
        num_nodes = x.size(0)
        
        if num_nodes == 1:
            # Single node case - no edges - on CPU
            edge_index = torch.zeros((2, 0), dtype=torch.long, device='cuda')
        else:
            # Create fully connected graph - on CPU
            node_indices = torch.arange(num_nodes, device='cuda')
            edge_index = torch.combinations(node_indices, r=2).t()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        return Data(x=x, edge_index=edge_index).to('cuda')
        
    def forward(self, persona_graph, story_graphs, positive_idx=None, 
                input_ids=None, attention_mask=None, labels=None,
                references=None, persona_sentences=None, story_sentences_list=None):
        """
        Joint forward pass with consistency classifier
        """
        # 1. Retrieval step - get similarity scores (on primary device)
        similarity_scores = self.retriever(persona_graph, story_graphs)
        
        # 2. Apply consistency classifier if sentences are provided
        consistency_scores = []
        if persona_sentences is not None and story_sentences_list is not None:
            try:
                # Build fully connected persona graph for consistency classifier - on CPU
                fc_persona_graph = self.build_fully_connected_graph(persona_sentences).to('cuda')
                
                # Apply classifier with all GNN operations on CPU
                for i, story_sentences in enumerate(story_sentences_list):
                    try:
                        # Build fully connected story graph - on CPU
                        fc_story_graph = self.build_fully_connected_graph(story_sentences)
                        
                        # Apply classifier with manual dimension handling - all on CPU
                        with torch.no_grad():
                            # Get persona embedding (64-dim)
                            h_p = self.consistency_classifier.gnn_persona(fc_persona_graph)
                            
                            # Get story embedding (64-dim) and expand to 128-dim
                            h_s_original = self.consistency_classifier.gnn_story(fc_story_graph)
                            h_s = self.story_expansion(h_s_original)  # Expand to 128 dimensions
                            
                            # Concatenate to 192 dimensions (64+128)
                            h_concat = torch.cat([h_p, h_s], dim=-1)
                            
                            # Pass directly to classifier layers
                            logits = self.consistency_classifier.classifier(h_concat)
                            score = torch.sigmoid(logits)
                        
                        consistency_scores.append(score.item())
                    except Exception as e:
                        print(f"Error processing story {i}: {e}")
                        # Fallback to neutral score
                        consistency_scores.append(0.5)
                
                # Move back to primary device
                consistency_scores = torch.tensor(consistency_scores, device=similarity_scores.device)
            except Exception as e:
                print(f"Error in consistency classifier: {e}")
                print("Falling back to similarity scores only")
                consistency_scores = torch.ones_like(similarity_scores) * 0.5
        else:
            # If sentences not provided, use default scores
            consistency_scores = torch.ones_like(similarity_scores) * 0.5
        
        # 3. Combine similarity and consistency scores
        # Hard filtering approach
        valid_stories = consistency_scores >= self.consistency_threshold
        if valid_stories.any():
            # Filter to only consider consistent stories
            filtered_scores = similarity_scores.clone()
            filtered_scores[~valid_stories] = -float('inf')
            combined_scores = filtered_scores
        else:
            # If no stories meet threshold, fall back to similarity
            combined_scores = similarity_scores
        
        # 4. Select story based on combined scores
        if self.training:
            story_sample = F.gumbel_softmax(combined_scores, tau=1.0, hard=True)
            selected_idx = torch.argmax(story_sample).item()
        else:
            selected_idx = torch.argmax(combined_scores).item()
            
        selected_story = story_graphs[selected_idx]
        
        # 5. Encode selected story graph for the generator
        story_embedding = self.retriever.encode_story(selected_story)
        text_embedding = self.graph_to_text(story_embedding)
        
        # 6. Generate with the graph-enhanced input
        encoder_outputs = self.prepare_encoder_with_graph(input_ids, text_embedding)
        
        outputs = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            encoder_outputs=encoder_outputs,
            output_hidden_states=True
        )
        
        # 7. Compute standard generation loss
        generation_loss = outputs.loss
        
        # Combined loss starts with generation loss
        
        
        # 8. Add retrieval loss if in training
        retrieval_loss = None
        if positive_idx is not None:
            retrieval_loss = self.retriever.compute_contrastive_loss(
                similarity_scores, positive_idx)
            
        
        # 9. Add BERTScore feedback if references provided
        bertscore_loss = None
        if references is not None:
            # Generate text (without teacher forcing)
            with torch.no_grad():
                generated_ids = self.generator.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    encoder_outputs=encoder_outputs,
                    max_length=100
                )
                
                # Decode generated texts
                generated_texts = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True)
                
            # Calculate BERTScore
            bertscore_loss = self.compute_bertscore_loss(generated_texts, references)
            
            # Add to total loss
            total_loss = generation_loss + self.lambda_retrieval * retrieval_loss + self.lambda_bertscore * bertscore_loss
            #import pdb; pdb.set_trace()
        # Update the output loss
        outputs.loss = total_loss
        
        # Return a dictionary with all components
        return {
            'loss': total_loss,
            'generation_loss': generation_loss,
            'retrieval_loss': retrieval_loss,
            'bertscore_loss': bertscore_loss,
            'logits': outputs.logits,
            'encoder_outputs': encoder_outputs
        }
    
    def prepare_encoder_with_graph(self, input_ids, graph_embedding):
        """Prepare T5 encoder outputs enhanced with graph embedding"""
        # Get original encoder outputs
        encoder_outputs = self.generator.encoder(input_ids)
        original_hidden_states = encoder_outputs.last_hidden_state
        
        batch_size = input_ids.shape[0]
        seq_len = original_hidden_states.size(1)
        hidden_dim = original_hidden_states.size(2)
        
        # Replace the first token instead of adding a new one
        graph_emb_expanded = graph_embedding.unsqueeze(0).expand(batch_size, 1, -1)
        enhanced_hidden_states = original_hidden_states.clone()
        enhanced_hidden_states[:, 0:1, :] = graph_emb_expanded
        
        # Update encoder outputs without changing sequence length
        encoder_outputs.last_hidden_state = enhanced_hidden_states
        return encoder_outputs
    
    def compute_bertscore_loss(self, candidates, references):
        """Compute BERTScore-based loss to encourage coherence"""
        # Tokenize candidates and references
        candidate_encodings = self.tokenizer(candidates, return_tensors="pt", 
                                           padding=True, truncation=True).to(self.bert_model.device)
        reference_encodings = self.tokenizer(references, return_tensors="pt", 
                                           padding=True, truncation=True).to(self.bert_model.device)
        
        # Get BERT embeddings
        with torch.no_grad():
            candidate_outputs = self.bert_model(**candidate_encodings)
            reference_outputs = self.bert_model(**reference_encodings)
            
            # Get token embeddings from last hidden state
            candidate_embeds = candidate_outputs.last_hidden_state
            reference_embeds = reference_outputs.last_hidden_state
            
            # Normalize embeddings
            candidate_embeds = F.normalize(candidate_embeds, p=2, dim=2)
            reference_embeds = F.normalize(reference_embeds, p=2, dim=2)
        
        # Compute cosine similarity matrix for each pair
        batch_scores = []
        for i in range(len(candidates)):
            # Get valid token embeddings (exclude padding)
            cand_mask = candidate_encodings.attention_mask[i].bool()
            ref_mask = reference_encodings.attention_mask[i].bool()
            
            c_embed = candidate_embeds[i, cand_mask]
            r_embed = reference_embeds[i, ref_mask]
            
            # Compute similarity matrix
            sim_matrix = torch.matmul(c_embed, r_embed.transpose(0, 1))
            
            # Compute precision, recall and F1
            precision = sim_matrix.max(dim=1)[0].mean()
            recall = sim_matrix.max(dim=0)[0].mean()
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            batch_scores.append(f1)
        
        # Average BERTScore F1 across batch
        avg_bertscore_f1 = torch.stack(batch_scores).mean()
        
        # Loss is 1 - F1 score (so higher F1 means lower loss)
        bertscore_loss = 1 - avg_bertscore_f1
        
        return bertscore_loss

    # def compute_bertscore_loss(self, candidates, references):
    #     cand_tok = self.tokenizer(candidates, padding=True, truncation=True,
    #                             return_tensors="pt")
    #     ref_tok  = self.tokenizer(references, padding=True, truncation=True,
    #                             return_tensors="pt")

    #     # Move to the BERT device
    #     cand_tok = {k: v.to(self.bert_model.device) for k, v in cand_tok.items()}
    #     ref_tok  = {k: v.to(self.bert_model.device) for k, v in ref_tok.items()}

    #     cand_h = self.bert_model(**cand_tok).last_hidden_state      # (B, Tc, d)
    #     ref_h  = self.bert_model(**ref_tok ).last_hidden_state      # (B, Tr, d)

    #     cand_h = F.normalize(cand_h, p=2, dim=-1)
    #     ref_h  = F.normalize(ref_h,  p=2, dim=-1)

    #     specials = set(self.tokenizer.all_special_ids)              # <pad>, <eos>, etc.
    #     batch_scores = []

    #     for i in range(len(candidates)):
    #         # valid tokens: mask==1 **and** not special
    #         c_valid = (cand_tok["attention_mask"][i] == 1) & \
    #                 (~torch.tensor([tid in specials for tid
    #                                 in cand_tok["input_ids"][i]], device=cand_h.device))
    #         r_valid = (ref_tok["attention_mask"][i] == 1) & \
    #                 (~torch.tensor([tid in specials for tid
    #                                 in ref_tok["input_ids"][i]], device=ref_h.device))

    #         c_embed = cand_h[i, c_valid]           # may be empty
    #         r_embed = ref_h[i, r_valid]

    #         if c_embed.numel() == 0 or r_embed.numel() == 0:
    #             batch_scores.append(torch.tensor(0.0, device=cand_h.device))
    #             continue

    #         sim = torch.matmul(c_embed, r_embed.T)
    #         precision = sim.max(dim=1)[0].mean()
    #         recall    = sim.max(dim=0)[0].mean()
    #         f1 = 2 * precision * recall / (precision + recall + 1e-8)
    #         batch_scores.append(f1)

    #     avg_f1 = torch.stack(batch_scores).mean()
    #     return 1 - avg_f1          # higher F1 → lower loss


def train_feedback_model(model, train_dataloader, optimizer, num_epochs):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in train_dataloader:
            # Unpack batch
            persona_graphs = batch['persona_graphs']
            story_graphs = batch['story_graphs']
            positive_indices = batch['positive_indices']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            references = batch['references']  # Reference responses
            
            # Forward pass with all components
            outputs = model(
                persona_graph=persona_graphs,
                story_graphs=story_graphs,
                positive_idx=positive_indices,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                references=references
            )
            
            loss = outputs['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Average loss: {total_loss/len(train_dataloader)}")

