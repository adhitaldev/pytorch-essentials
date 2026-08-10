
import torch.cuda as cuda
class TransformerConfig:
    """
    Basic configuration class for a basic transformer model.
    This contains the hyperparameters and settings for the transformers.
    """
    block_size = 128
    n_embd = 256
    n_head = 8
    n_layer = 6
    dropout = 0.1
    batch_size = 32
    learning_rate = 3e-4
    max_iters = 3000
    eval_interval = 300
    eval_iters = 50
    device = 'cuda' if cuda.is_available() else 'cpu'
    vocab_size = None

    def __str__(self):
        return f"""
                    Block_size: {self.block_size} NumEmbeddings{self.n_embd}
                    NumHeads: {self.n_head} NumLayers: {self.n_layer} Dropout: {self.dropout}
                    BatchSize: {self.batch_size} LRate: {self.learning_rate} MaxIters: {self.max_iters}
                    EvalInterval: {self.eval_interval} Device: {self.device} Vocab_size: {self.vocab_size}
                """
        
