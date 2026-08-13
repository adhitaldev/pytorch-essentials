"""
Describes the Transformer NN as and its composite modules
as defined in the paper 'Attention is All You Need' (Vaswant, et. all)
Implementation is as close as possible to the original work.
"""
import torch.nn

class Head(nn.Module):
    """
    Single Attention head with an attention function that maps a query and a set 
    of key-value pairs to an output.
    """
    def __init__(self, head_size, d_model, block_size, dropout):
        super().__init__()
        # nn.Linear creates a fully connected dense layer and applies
        # affine transformation/creates weight W and bias b as in y = xW^t + b
        # No bias in this case though.
        # With a d_model of 256 and head_size of 8, the original 256 sized token
        # embeddings are projected to a 32 dim vectors. Headsize is the output
        # dimension of thoose three projections.
        """ Example: 
        For a 3 token sentense with d_model = 4 and head_size = 2,
        suppose after projection we get, 
        Q = 
            "The" → q = [0.1, 0.9]
            "cat" → q = [0.8, 0.2]
            "sat" → q = [0.3, 0.7]
        K = 
            "The" → k = [0.2, 0.8]
            "cat" → k = [0.9, 0.3]
            "sat" → k = [0.4, 0.6]
        V = 
            "The" → v = [1.0, 0.0]
            "cat" → v = [0.0, 1.0]
            "sat" → v = [0.5, 0.5]
        -------
        For "sat", we dot against every key to find how close they are.
        This is the 'compatibility' func. of query and key (Section 3.2 in the paper)
        Using dot product because of highly optimized matrix multiplications.
        --------- 
        sat_query. The_key = 0.3*0.2 + 0.7*0.8 = 0.62
        sat_query. cat_key = 0.3*0.9 + 0.7*0.3 = 0.48
        sat_query. sat_key = 0.3*0.4 + 0.7*0.6 = 0.54
    
        ---------
        These get scaled by 1/sqrt(head_size). (d_k in section 3.2.1) and
        then we apply softmax to get the weights for each value.
        Thi scaling is done to avoid dot products going too large in magnitude,
        pushing the softmax function into regions where it has extremely small gradients.
        ---------
        sqrt_dk = sqrt(2) = 1.414
        Attention(Q, K, V) for "sat"= 
            softmax([0.62/sqrt_dk, 0.48/sqrt_dk, 0.54/sqrt_dk]) 
            = [0.36, 0.29, 0.35]

        Output of "sat" is weighted sum of values.
        output("sat") = weight("The") * v("The") + weight("cat") * v("cat") + weight("sat") * v("sat")
         = 0.36 * [1.0, 0.0] + 0.29 * [0.0, 1.0] + 0.35 * [0.5, 0.5]
         = [0.36, 0.0] + [0.0, 0.29] + [0.175, 0.175]
         = [0.535, 0.465]
        is "sat"'s new context aware representation.

        In the context for paper, three projections are used instead of a single
        embedding to separate out the match-relevance(query, key) from content
        derivation (value). The query and key are used to compute the attention
        weights and the value is used to compute the output as a weighted sum of values.
        """
        # nn.Linear(in_features/dims of input neurons, out_features/dims of output neurons, bias=True/False)
        self.key = nn.Linear(d_model, head_size, bias=False) # what do I contain for matching?
        self.query = nn.Linear(d_model, head_size, bias=False)# what am I looking for?
        self.value = nn.Linear(d_model, head_size, bias=False)# what can I offer, if selected?
        
        # Zero out certain percentage of the elements of the input tensor.
        self.dropout = nn.Dropout(dropout)

        # Creates a casual mask that stops each token from seeing future tokens
        # during attention, which makes it an autogressive model.
        # torch.ones (block_size, block_size) creates a square matrix of all 1s
        # torch.tril zeros out the everything above the main diagonal.
        # tril[i][j] where i is current token position and j is position being attended to.
        # Example : 
        #[[ 1, 1, 1
        #   1, 1, 1
        #   1, 1, 1
        # ]] === >
        # [[ 1, 0, 0
        #    1, 1, 0
        #    1, 1, 1
        # ]]
        # The third token (1, 1, 1) can attend to all three tokens, but the second token(1, 1, 0)
        # cannot attend to the third token.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # In the nn.module, the overwritten forward function defines 
        # the computation that this module performs on input data.
        # Vectorized implementation of the attention func. defined in section 3.2.1.
        # Attention (Q, K, V) = softmax(QK^T/sqrt(d_k))V
        def forward(self, x):
            # B = Batch Size, T = Time Steps/Sequence Length, C = Channels/Embedding dimensions/Features
            B, T, C = x.shape
            k = self.key(x) # (B, T, head_size)
            q = self.query(x)
            # QK^T transpose the last two dimensions of k
            # turning (B, T, head_size) into (B, head_size, T)
            # C** -0.5 is the 1/sqrt(d_k) scaling factor
            wei = q @ k.transpose(-2, -1) * C**-0.5

            # Casual masking
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

            # Apply softmax
            wei = torch.nn.functional.softmax(wei, dim=-1)

            # Apply dropout
            wei = self.dropout(wei)

            v = self.value(x)
            out = wei @ v # (B, T, head_size)
            return out

class Block(nn.Module):
    """
    A single transofrmer block that consists of the 
    attention + feed forward each with a residual connection
    and a pre-LayerNorm.
    """
    def __init__(self, d_model, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.attention = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)


class TransformerLM(nn.Module):
    """
    The full model.
    Notes: The dims of token embedding and positional embedding must match as they are added together.
    """
    def __init__(self, vocab_size, d_model, block_size, n_head, n_layer, dropout):
        super().__init__()
        # Size of the context window
        self.block_size = block_size
        # torch.nn.Embedding describes the lookup table for 
        # the embedding vector for each token in the vocabulary
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # The positional embedding is a lookup table for the
        # position of each token in the context window (block_size)
        # Where could each token be in the context window?
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential()




