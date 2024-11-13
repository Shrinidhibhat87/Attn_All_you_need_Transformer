# Importing PyTorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import Standard libraries
import math


# A function to get the Scaled-dot product attention
def scaled_dot_product_attention(query, key, value, mask=None) -> torch.Tensor:
    """
    Calculate the scaled dot-product attention.
    Args:
        query (torch.Tensor): The query matrix of shape (..., seq_len_q, depth).
        key (torch.Tensor): The key matrix of shape (..., seq_len_k, depth).
        value (torch.Tensor): The value matrix of shape (..., seq_len_v, depth_v).
        mask (torch.Tensor, optional): The mask matrix of shape (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
        torch.Tensor: The output tensor after applying scaled dot-product attention.
        torch.Tensor: The attention weights before multiplying with values.
    """
    # Get the depth for scaling down the dot product
    d_k = query.size(-1)

    # Calculate the dot product between query and key
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply the mask (if any)
    # The mask helps to avoid attending to the padding tokens
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
    
    # Apply the softmax activation
    attention_weights = F.softmax(attention_scores, dim=-1)

    # Calculate the final attention output
    attention_output = torch.matmul(attention_weights, value)

    return attention_output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Implements the multi-head attention mechanism as described in the 'Attention Is All You Need' paper.
    Multi-head attention allows the model to jointly attend to information from different representation
    subspaces at different positions.

    Attributes:
        embed_dim (int): The total embedding dimension of the input.
        num_heads (int): The number of attention heads. Each head computes its attention with a 
            separate set of weights.
        head_dim (int): The dimension of each head, which is computed as embed_dim // num_heads.
        query (nn.Linear): Linear layer for generating query vectors.
        key (nn.Linear): Linear layer for generating key vectors.
        value (nn.Linear): Linear layer for generating value vectors.
        out (nn.Linear): Linear layer for the output after concatenating attention heads.
    """
    def __init__(self, embed_dim, num_heads):
        """
        Initializes the MultiHeadAttention layer.

        Args:
            embed_dim (int): The dimension of the input embeddings.
            num_heads (int): The number of attention heads to use.

        Raises:
            AssertionError: If embed_dim is not divisible by num_heads.
        """
        # Call the parent class constructor
        super(MultiHeadAttention, self).__init__()

        # Because we are using multi-head attention, we need to split the embedding dimension into multiple heads
        # Check if the dimension is divisible by the number of heads
        assert embed_dim % num_heads == 0, "Embedding dimension is not divisible by the number of heads"

        # Assign the input parameters
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Calculate the dimension of each head
        self.head_dim = embed_dim // num_heads

        # Define the query, key and value linear layers
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Define the output linear layer
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        """
        Performs forward pass through the multi-head attention mechanism.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, sequence_length, embed_dim).
            key (torch.Tensor): Key tensor of shape (batch_size, sequence_length, embed_dim).
            value (torch.Tensor): Value tensor of shape (batch_size, sequence_length, embed_dim).
            mask (torch.Tensor, optional): Optional mask tensor to prevent attention to certain positions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - attention_output: The output tensor of shape (batch_size, sequence_length, embed_dim).
                - attention_weights: Attention weights for each head.
        """
        # Get the batch size and sequence length
        batch_size, sequence_length, _ = query.size()

        # Apply the linear layers to get the query, key and value matrices
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Reshape the query, key and value matrices to split the embedding dimension into multiple heads
        query = query.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        key = key.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        value = value.view(batch_size, sequence_length, self.num_heads, self.head_dim)

        # Transpose the query, key and value matrices to prepare for the scaled dot-product attention
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Apply the scaled dot-product attention
        attention_output, attention_weights = scaled_dot_product_attention(query, key, value, mask)

        # Combine heads and project to output
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.embed_dim)
        return self.out(attention_output), attention_weights

class PositionWiseFeedForward(nn.Module):
    """
    Implements the Position-wise Feedforward Network from 'Attention Is All You Need'.
    This layer applies two fully connected (FC) layers with a ReLU activation in between.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer which maps input embeddings to a higher-dimensional space.
        fc2 (nn.Linear): The second fully connected layer which projects back to the original embedding dimension.
        relu (nn.ReLU): ReLU activation function applied between the two layers.
    """
    def __init__(self, embed_dim, inner_dim) -> None:
        """
        Initializes the PositionWiseFeedForward layer.

        Args:
            embed_dim (int): The dimension of the input and output embeddings.
            inner_dim (int): The dimension of the hidden layer (typically larger than embed_dim).
        """
        # Call the parent class constructor
        super(PositionWiseFeedForward, self).__init__()

        # Based on the original paper, there are two fully connected layers with ReLU activation in between
        # First fully connected layer: Projects input from embed_dim to inner_dim
        self.fc1 = nn.Linear(embed_dim, inner_dim)
        # Second fully connected layer: Projects from inner_dim back to embed_dim
        self.fc2 = nn.Linear(inner_dim, embed_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Applies the position-wise feedforward network to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, embed_dim),
                          with the same dimensions as input after the transformation.
        """
        # Apply the first FC layer
        x = self.fc1(x)
        # Non-linear ReLU activation
        x = self.relu(x)
        # Apply the second FC layer
        x = self.fc2(x)

        return x

class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings using sine and cosine functions
    of different frequencies, allowing the model to identify the position of tokens within a sequence.

    Attributes:
        position_encodings (torch.Tensor): A fixed positional encoding matrix of shape 
                                           (1, max_len, embed_dim), added to input embeddings.
    """
    def __init__(self, embed_dim, max_len=1000):
        """
        Initializes the PositionalEncoding layer.

        Args:
            embed_dim (int): The dimension of the embeddings.
            max_len (int, optional): The maximum sequence length to support. Default is 1000.
        """
        # Call the parent class constructor
        super(PositionalEncoding, self).__init__()

        # Create a matrix of shape (max_len, embed_dim) to store the positional encodings
        position_encodings = torch.zeros(max_len, embed_dim)

        # Generate positions from 0 to max_len and calculate their encodings
        positions = torch.arange(0, max_len).unsqueeze(1).float()

        # Compute the scaling factor for the frequency terms in positional encoding
        denominator = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))

        # Apply sine to even indices and cosine to odd indices in embedding dimension
        position_encodings[:, 0::2] = torch.sin(positions * denominator) # even indices
        position_encodings[:, 1::2] = torch.cos(positions * denominator) # odd indices

        # Register positional encodings as a non-trainable buffer
        self.register_buffer('position_encodings', position_encodings.unsqueeze(0))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embed_dim).

        Returns:
            torch.Tensor: Output tensor of the same shape as input, with positional encodings added.
        """
        x = x + self.position_encodings[:, :x.size(1)]
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Implements a single layer of the Transformer encoder as described in the 'Attention Is All You Need' paper.

    Attributes:
        self_attn (MultiHeadAttention): Multi-head self-attention mechanism.
        feed_forward (PositionWiseFeedForward): Position-wise feedforward network.
        norm1 (nn.LayerNorm): Layer normalization for the self-attention output.
        norm2 (nn.LayerNorm): Layer normalization for the feedforward network output.
        dropout (nn.Dropout): Dropout layer applied to the output of each sub-layer.
    """
    def __init__(self, embed_dim, num_heads, inner_dim, dropout=0.1):
        """
        Initializes the TransformerEncoderLayer.

        Args:
            embed_dim (int): The input and output dimension of the embeddings.
            num_heads (int): The number of attention heads to use.
            inner_dim (int): The dimension of the inner layer in the position-wise feedforward network.
            dropout (float, optional): The dropout rate to apply. Defaults to 0.1.
        """
        # Call the parent class constructor
        super(TransformerEncoderLayer, self).__init__()

        # The below definition is based on the original paper
        # Self attention layer
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        # Feedforward network
        self.feed_forward = PositionWiseFeedForward(embed_dim, inner_dim)
        # Layer normalization for the self-attention output
        self.norm1 = nn.LayerNorm(embed_dim)
        # Layer normalization for the feedforward network output
        self.norm2 = nn.LayerNorm(embed_dim)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x:torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:
        """
        Forward pass of the Transformer encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embed_dim).
            mask (torch.Tensor, optional): Optional mask tensor to prevent attention to certain positions.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, embed_dim).
        """
        # Apply self-attention mechanism
        attention_output = self.self_attn(x, x, x, mask)
        # Apply dropout to the attention output
        x = x + self.dropout(attention_output)
        # Apply layer normalization
        x = self.norm1(x)

        # Apply position-wise feedforward network
        feed_forward_output = self.feed_forward(x)
        # Apply dropout to the feedforward output
        x = x + self.dropout(feed_forward_output)
        # Apply layer normalization
        x = self.norm2(x)

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, inner_dim, dropout=0.1):
        """
        Initializes the TransformerDecoderLayer.

        Args:
            embed_dim (int): The input and output dimension of the embeddings.
            num_heads (int): The number of attention heads to use.
            inner_dim (int): The dimension of the inner layer in the position-wise feedforward network.
            dropout (float, optional): The dropout rate to apply. Defaults to 0.1.
        """
        # Call the parent class constructor
        super(TransformerDecoderLayer, self).__init__()

        # Define the self-attention and cross attention mechanisms
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)

        # Define the position-wise feedforward network
        self.feed_forward = PositionWiseFeedForward(embed_dim, inner_dim)

        # Define the layer normalization layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        # Define the dropout layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        """
        Forward pass of the Transformer decoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, target_length, embed_dim).
            enc_out (torch.Tensor): The output of the encoder stack of shape (batch_size, source_length, embed_dim).
            self_mask (torch.Tensor, optional): Optional mask tensor to prevent attending to future positions.
            cross_mask (torch.Tensor, optional): Optional mask tensor to prevent attending to padding tokens in the source.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, target_length, embed_dim).
        """
        # Apply self-attention mechanism
        self_attention_output = self.self_attn(x, x, x, self_mask)
        # Apply dropout to the self-attention output
        x = x + self.dropout(self_attention_output)
        # Apply layer normalization
        x = self.norm1(x)

        # Apply cross-attention mechanism
        cross_attention_output = self.cross_attn(x, enc_out, enc_out, cross_mask)
        # Apply dropout to the cross-attention output
        x = x + self.dropout(cross_attention_output)
        # Apply layer normalization
        x = self.norm2(x)

        # Apply position-wise feedforward network
        feed_forward_output = self.feed_forward(x)
        # Apply dropout to the feedforward output
        x = x + self.dropout(feed_forward_output)
        # Apply layer normalization
        x = self.norm3(x)

        return x


class Transformer(nn.Module):
    """
    Implements a Transformer model with encoder-decoder architecture based on 'Attention Is All You Need'.

    Attributes:
        encoder (nn.ModuleList): Stack of encoder layers.
        decoder (nn.ModuleList): Stack of decoder layers.
        positional_encoding (PositionalEncoding): Adds positional encodings to embeddings.
        encoder_embedding (nn.Linear): Embedding layer for encoder inputs.
        decoder_embedding (nn.Linear): Embedding layer for decoder inputs.
        out (nn.Linear): Output projection layer to produce final output embeddings.
        dropout (nn.Dropout): Dropout layer for regularization.
    """
    def __init__(self, num_encoder_layers, num_decoder_layers, embed_dim, num_heads, inner_dim, dropout=0.1):
        """
        Initializes the Transformer model.

        Args:
            num_encoder_layers (int): Number of layers in the encoder stack.
            num_decoder_layers (int): Number of layers in the decoder stack.
            embed_dim (int): Dimension of embeddings for input and output.
            num_heads (int): Number of attention heads.
            inner_dim (int): Dimension of the inner layer in the feedforward network.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(Transformer, self).__init__()

        # Encoder and decoder layer stacks
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, inner_dim, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, inner_dim, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Positional encoding for inputs
        self.positional_encoding = PositionalEncoding(embed_dim)

        # Embedding layers for encoder and decoder inputs
        self.encoder_embedding = nn.Linear(embed_dim, embed_dim)
        self.decoder_embedding = nn.Linear(embed_dim, embed_dim)

        # Output layer to project the final decoder output to target vocabulary
        self.out = nn.Linear(embed_dim, embed_dim)

        # Dropout for embeddings
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Defines the forward pass for the Transformer model.

        Args:
            src (torch.Tensor): Source sequence input tensor of shape (batch_size, src_seq_length, embed_dim).
            tgt (torch.Tensor): Target sequence input tensor of shape (batch_size, tgt_seq_length, embed_dim).
            src_mask (torch.Tensor, optional): Mask tensor for the source input.
            tgt_mask (torch.Tensor, optional): Mask tensor for the target input (e.g., to prevent attending to future tokens).
            memory_mask (torch.Tensor, optional): Mask tensor for encoder-decoder attention (optional).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, tgt_seq_length, embed_dim).
        """
        # Apply embeddings and positional encoding for source input
        src = self.encoder_embedding(src)
        src = self.dropout(self.positional_encoding(src))
        
        # Pass source through encoder layers
        memory = src
        for layer in self.encoder:
            memory = layer(memory, src_mask)
        
        # Apply embeddings and positional encoding for target input
        tgt = self.decoder_embedding(tgt)
        tgt = self.dropout(self.positional_encoding(tgt))

        # Pass target through decoder layers, using memory from encoder as context
        output = tgt
        for layer in self.decoder:
            output = layer(output, memory, tgt_mask, memory_mask)

        # Project output to the embedding dimension
        output = self.out(output)
        return output


def main():
    mh_attention = MultiHeadAttention(512, 8)
    query = torch.randn(64, 10, 512)
    key = torch.randn(64, 10, 512)
    value = torch.randn(64, 10, 512)
    output, attention_weights = mh_attention(query, key, value)
    print(output.shape)


if __name__ == "__main__":
    main()