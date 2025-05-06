import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from GraphRAG_train import BERTScoreFeedbackRAG, EnhancedBERTScoreFeedbackRAG


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 
                   'cpu')
print(f"Using device: {device}")

# Data paths
train_data_path = "/Users/tzhang/projects/Personalized_RAG/data/LAPDOG_new_dataset/train_data_10000.jsonl"
story_data_path = "/Users/tzhang/projects/Personalized_RAG/data/story.jsonl"

# Load data
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        i = 0
        for line in f:
            data.append(json.loads(line))
            i += 1
            if i > 100:
                break
    return data

train_data = load_jsonl(train_data_path)
story_data = load_jsonl(story_data_path)

print(f"Loaded {len(train_data)} train examples and {len(story_data)} stories")

# Initialize the SentenceTransformer model for embeddings
sentence_model = SentenceTransformer('Lajavaness/bilingual-embedding-small', device=device,trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('t5-small')

# Function to construct graph from text elements
def construct_graph(text_elements):
    """
    Construct a graph from text elements (persona attributes or story sentences)
    Returns a PyG Data object
    """
    # Encode text elements
    embeddings = sentence_model.encode(text_elements, convert_to_tensor=True)
    
    # Compute similarity
    embeddings_np = embeddings.cpu().numpy()
    similarity_matrix = cosine_similarity(embeddings_np)
    
    # Create edges based on threshold
    threshold = 0.7
    edge_index = [[], []]
    num_nodes = len(text_elements)
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if similarity_matrix[i][j] >= threshold:
                edge_index[0].extend([i, j])
                edge_index[1].extend([j, i])
    
    # Create PyG Data object
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    x = torch.tensor(embeddings_np, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index)

# Process stories and create story graphs
print("Processing stories into graphs...")
story_graphs = []
for story in tqdm(story_data):
    # Split story into sentences
    sentences = story['text'].split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    
    # Construct graph
    if len(sentences) > 1:  # Ensure at least 2 nodes for a graph
        story_graph = construct_graph(sentences)
        story_graphs.append(story_graph)

print(f"Created {len(story_graphs)} story graphs")

def extract_persona(text):
    """
    Extract persona attributes from text that contains both persona and context.
    Returns a list of persona statements.
    """
    # Split the text into sections
    parts = text.split("context:")
    
    if len(parts) < 2:
        return []  # No context section found
    
    # Get the persona part and remove the "persona:" prefix
    persona_part = parts[0].strip()
    if persona_part.startswith("persona:"):
        persona_part = persona_part[len("persona:"):].strip()
    
    # Split into individual statements
    persona_statements = [stmt.strip() for stmt in persona_part.split('.') if stmt.strip()]
    
    return persona_statements

def extract_context(text):
    """
    Extract dialogue context from text that contains both persona and context.
    Returns a list of dialogue turns.
    """
    # Split the text into sections
    parts = text.split("context:")
    
    if len(parts) < 2:
        return []  # No context section found
    
    # Get the context part
    context_part = parts[1].strip()
    
    # Split into Q/R turns
    turns = []
    
    # Replace common markers with standardized ones to make splitting easier
    context_part = context_part.replace("Q:", "Q: ").replace("R:", "R: ")
    
    # Split by Q: for questions
    q_splits = context_part.split("Q: ")
    
    # Skip the first element if it's empty
    q_splits = [s for s in q_splits if s.strip()]
    
    for q_part in q_splits:
        # Split each Q part into question and response
        r_splits = q_part.split("R: ")
        
        if len(r_splits) > 0:
            # First part is the question
            question = r_splits[0].strip()
            turns.append(f"Q: {question}")
            
            # If there's a response
            if len(r_splits) > 1:
                response = r_splits[1].strip()
                turns.append(f"R: {response}")
    
    return turns


# Custom Dataset
class PersonaDialogueStoryDataset(Dataset):
    def __init__(self, data, story_data, story_graphs, tokenizer, max_length=512):
        self.data = data
        self.story_data = story_data  # Add the original story data
        self.story_graphs = story_graphs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Store story sentences for quick access
        self.story_sentences = []
        for story in story_data:
            sentences = story['text'].split('. ')
            sentences = [s.strip() + '.' for s in sentences if s.strip()]
            self.story_sentences.append(sentences)
        
        # Pre-compute persona graphs
        print("Processing personas into graphs...")
        self.persona_graphs = []
        for item in tqdm(data):
            question = item['question']
            persona = extract_persona(question)
            if len(persona) > 1:  # Ensure at least 2 nodes
                persona_graph = construct_graph(persona)
                self.persona_graphs.append(persona_graph)
            else:
                # Fallback for short personas: duplicate the attribute
                temp_persona = persona + persona  # Duplicate to ensure graph creation
                persona_graph = construct_graph(temp_persona)
                self.persona_graphs.append(persona_graph)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        persona_graph = self.persona_graphs[idx]
        
        # Extract persona information
        question = item['question']
        persona = extract_persona(question)
        
        # For positive story index, use similarity-based selection initially
        positive_idx = self.get_initial_positive_story(persona)
        
        # Prepare input text (dialogue context)
        history = extract_context(question)
        input_text = ' '.join(history[-6:])  # Use last 6 turns as context
        
        # Tokenize input and target
        input_encoding = self.tokenizer(input_text, padding='max_length', 
                                       truncation=True, max_length=self.max_length, 
                                       return_tensors='pt')
        
        target_text = item['answers'][0]
        target_encoding = self.tokenizer(target_text, padding='max_length', 
                                        truncation=True, max_length=self.max_length, 
                                        return_tensors='pt')
        
        return {
            'persona_graph': persona_graph,
            'positive_idx': positive_idx,
            'input_ids': input_encoding.input_ids.squeeze(),
            'attention_mask': input_encoding.attention_mask.squeeze(),
            'labels': target_encoding.input_ids.squeeze(),
            'reference': target_text,
            'persona_sentences': persona,
            'dialogue_context': history,
        }
    
    def get_initial_positive_story(self, persona):
        """Select initial positive story based on similarity"""
        # Combine persona into a single text
        persona_text = ' '.join(persona)
        
        # Get embedding - initially on CPU
        persona_emb = sentence_model.encode(persona_text, convert_to_tensor=True)
        
        # Sample a subset of stories for efficiency
        sample_size = min(100, len(self.story_graphs))
        sample_indices = np.random.choice(len(self.story_graphs), sample_size, replace=False)
        
        best_similarity = -1
        best_idx = 0
        
        # Use CPU for all similarity comparisons to avoid device mismatch
        for i in sample_indices:
            story_graph = self.story_graphs[i]
            # Compute mean embedding and move to CPU
            story_emb = torch.mean(story_graph.x, dim=0).cpu()
            
            # Ensure persona embedding is also on CPU
            persona_emb_cpu = persona_emb.cpu()
            
            # Compute similarity on CPU
            similarity = torch.cosine_similarity(
                persona_emb_cpu.unsqueeze(0), 
                story_emb.unsqueeze(0)
            ).item()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = i
                
        return best_idx

# Create dataset and dataloader
dataset = PersonaDialogueStoryDataset(train_data, story_data, story_graphs, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: x)

# Initialize the model
model = EnhancedBERTScoreFeedbackRAG(
    hidden_dim=128, 
    out_dim=64,
    classifier_path="/Users/tzhang/projects/Personalized_RAG/model/classifer/consistency_classifier_fold5.pt"
).to(device)

# Initialize optimizer
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
num_epochs = 5
save_path = "graphrag_model"

print("Starting training...")
model.train()

for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_gen_loss = 0
    epoch_retrieval_loss = 0
    epoch_bertscore = 0
    step_count = 0
    
    progress_bar = tqdm(dataloader)
    for step, batch in enumerate(progress_bar):
        # Process each example in the batch individually
        batch_loss = 0
        batch_gen_loss = 0
        batch_retrieval_loss = 0
        batch_bertscore_loss = 0
        
        for item in batch:
            # Move tensors to device
            persona_graph = item['persona_graph'].to(device)
            input_ids = item['input_ids'].to(device).unsqueeze(0)
            attention_mask = item['attention_mask'].to(device).unsqueeze(0)
            labels = item['labels'].to(device).unsqueeze(0)
            positive_idx = item['positive_idx']
            reference = [item['reference']]
            
            # Sample story graphs for efficiency
            sample_size = min(50, len(story_graphs))
            sample_indices = np.random.choice(len(story_graphs), sample_size, replace=False)
            
            
            # Ensure positive story is included
            if positive_idx not in sample_indices:
                sample_indices[0] = positive_idx
                
            # Get the position of positive index in the sampled indices
            pos_idx_in_sample = np.where(sample_indices == positive_idx)[0][0]
            
            # Get sampled story graphs
            sampled_story_graphs = [story_graphs[i].to(device) for i in sample_indices]
            
            # Get the corresponding story sentences from the dataset
            sampled_story_sentences = [dataset.story_sentences[i] for i in sample_indices]
            
            # Forward pass
            outputs = model(
                persona_graph=persona_graph,
                story_graphs=sampled_story_graphs,
                positive_idx=pos_idx_in_sample,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                references=reference,
                persona_sentences=item['persona_sentences'],
                story_sentences_list=sampled_story_sentences
            )
            
            # Get loss
            loss = outputs['loss']
            
            # ADD THESE LINES FOR BACKPROPAGATION
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Get loss components directly
            total_loss = outputs['loss']
            generation_loss = outputs['generation_loss']
            retrieval_loss = outputs['retrieval_loss']
            bertscore_loss = outputs['bertscore_loss']
            
        
        # Accumulate losses for tracking
        batch_loss += total_loss.item()
        batch_gen_loss += generation_loss.item()
        batch_retrieval_loss += retrieval_loss.item() * model.lambda_retrieval
        batch_bertscore_loss += bertscore_loss.item()
        step_count += 1
        
        # Update progress bar
        progress_bar.set_description(
            f"Epoch {epoch+1} | Loss: {batch_loss:.4f} | BERTScore: {batch_bertscore_loss:.4f}"
        )
        
        # Print detailed metrics every 10 steps
        if step % 10 == 0:
            print(f"\nStep {step} | Total Loss: {batch_loss:.4f} | Gen Loss: {batch_gen_loss:.4f} | "
                  f"Retrieval Loss: {batch_retrieval_loss:.4f} | BERTScore: {batch_bertscore_loss:.4f}\n")
    
    # Epoch average metrics
    epoch_loss = batch_loss / step_count
    epoch_gen_loss = batch_gen_loss / step_count
    epoch_retrieval_loss = batch_retrieval_loss / step_count
    epoch_bertscore_loss = batch_bertscore_loss / step_count
    
    print(f"Epoch {epoch+1}/{num_epochs} Summary:")
    print(f"  Average Loss: {epoch_loss:.4f}")
    print(f"  Average Gen Loss: {epoch_gen_loss:.4f}")
    print(f"  Average Retrieval Loss: {epoch_retrieval_loss:.4f}")
    print(f"  Average BERTScore Loss: {epoch_bertscore_loss:.4f}")
    
    # Save model after each epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }, f"{save_path}_epoch_{epoch+1}.pt")

print("Training complete!")

