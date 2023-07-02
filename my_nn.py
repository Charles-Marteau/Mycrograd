# A selection of handmade layers which act on our Tnsr class
# We only use the numpy package functionalities

# - Linear Layer
# - Feed forward (successive linear layers with ReLU activations in between)
# - Softmax
# - Cross entropy
# - Positional encoding (to be added to the embedding in the transformer)
# - Layer norm
# - Masked Attention 
# - Masked multi_head attention
# - Sequence
# - Embedding
# - Decoder block
# - Transformer


from my_engine import Tnsr
import numpy as np



class Module:
    """
    Methods that are inherited by all the layers.
    """
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data, dtype=p.data.dtype)

    def parameters(self):
        return []

class Linear(Module):
    """
    Linear layer with input dimension dim_in, output dimension
    dim_out and ReLu non-linearity which is turned on if
    nonlin = True.
    """
    def __init__(self, dim_in, dim_out, nonlin=True, bias=True):
        super().__init__()
        self.w = Tnsr(np.random.randn(dim_out, dim_in) / np.sqrt(dim_in))
        self.b = Tnsr(np.random.randn(dim_out))
        self.nonlin = nonlin
        self.bias = bias

    def __call__(self, x):
        if self.bias==True:
            output = x.matmul(self.w.transpose(-1, -2)) + self.b
        else:
            output = x.matmul(self.w.transpose(-1, -2))
        return output.relu() if self.nonlin == True else output

    def parameters(self):
        return [self.w, self.b] if self.bias == True else [self.w]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron: \
                   dim_in = {self.dim_in}, dim_out = {self.dim_out} "
    
class FeedForward(Module):
    """
    Multi layer perceptron
    dims: list of integers [dim_in, hidden_1, ..., hidden_n, dim_out]
    No ReLu in the last layer
    """
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.layers = [Linear(dims[i], dims[i+1], nonlin=(i < len(dims)-2)) 
                        for i in range(len(dims)-1)]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params
    
    def __repr__(self):
        return 'FeedForward:' + ' ' + ' '.join(
            [f'dim_in = {self.dims[0]}',
            *[f'hidden_{k} = {self.dims[k]}'
               for k in range(1, len(self.dims) - 1)],
            f'dim_out = {self.dims[-1]}']
            )

class softmax(Module):
    """
    Softmax layer with regularization by substraction of 
    the max.
    """
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def __call__(self, x):
        max_x = x.max(axis = self.axis, keepdims=True)
        soft =  (x - max_x).exp() / (
            x - max_x).exp().sum(axis = self.axis, keepdims=True)
        return soft
    
class cross_entropy(Module):    
    """
    logits = (n_mini_batch, n_classes) tensor (un-normalized logit),
    classes = (n_minibatch,) tensor
    """
    def __init__(self):
        super().__init__()

    def __call__(self, logits, classes):
        def one_hot(classes, n_classes):
            X = np.arange(n_classes)
            Y = np.expand_dims(classes, axis=-1)
            return (X == Y).astype(np.float64)
        y_one_hot = Tnsr(one_hot(classes.data, logits.data.shape[-1]))
        return - (
            y_one_hot * (softmax(-1)(logits).log())
            ).sum() / np.float64(classes.data.size)
    
    
class pos_encoding(Module):
    """positional encoding: N controls the frequencies, d_model is the input
       size dimension"""
    def __init__(self, N, d_model):
        super().__init__()
        self.N = N
        self.d_model = d_model

    def __call__(self, seq_length):
        """returns a dim (seq_length, d_model) array"""
        position = np.zeros((seq_length, self.d_model))
        t = np.arange(seq_length)
        for k in range(0, int(np.floor(self.d_model / 2)) + 1):
            omega = 1 / self.N**(2*((k + 1)/self.d_model))
            if 2*k <= self.d_model - 1:
                position[:, 2*k] = np.cos(t*omega)
            if 2*k + 1 <= self.d_model - 1:
                position[:, 2*k + 1] = np.sin(t*omega)
        return Tnsr(position)
    
    
class LayerNorm(Module):
    """
    Defines a layer norm along axis of size axis_size   
    """
    def __init__(self, axis, axis_size):
        super().__init__()
        self.axis = axis
        self.axis_size = axis_size
        self.gamma = Tnsr(np.array(1, dtype=np.float64))
        self.beta = Tnsr(
            np.zeros(self.axis_size, dtype=np.float64))

    def __call__(self, x):  
        eps = Tnsr(np.float64(1e-05))
        mean = x.mean(axis=self.axis, keepdims=True)
        sigma = (
            ((x - mean) ** 2).mean(axis=self.axis, keepdims=True) + eps
            ).sqrt()
        if x.data.shape[-1] == 1:
            return x * self.gamma + self.beta
        else:
            return ((x - mean) / sigma) * self.gamma + self.beta
        
    def parameters(self):
        return [self.gamma, self.beta]
    
    def __repr__(self):
        return f'LayerNorm: axis = {self.axis}'
    
class masked_attention(Module):
    """
    Args:
        - q: Tnsr(batch_size, num_heads, seq_len, headsize)
        - k: Tnsr(batch_size, num_heads, seq_len, headsize)
        - v: Tnsr(batch_size, num_heads, seq_len, headsize)
    Returns:
        - out: Tnsr(batch_size, num_heads, seq_len, headsize)
    """
    # att_matrix: Tnsr(batch_size, num_heads, seq_len, seq_len)
    def __init__(self):
        super().__init__()
        
    def __call__(self, q, k, v):
        att_matrix = q.matmul(k.transpose(-1, -2))
        headsize = q.data.shape[-1]
        att_matrix = att_matrix / Tnsr(headsize).sqrt()
        att_matrix = att_matrix.mask()
        weights = softmax(-1)(att_matrix)
        return weights.matmul(v)
    

class masked_multi_head_attention(Module):
    """
    Defines a masked multi head attention that acts on an input a tensor 
    x: Tnsr(batch_size, seq_len, d_model) and ouputs a tensor 
    out: Tnsr(batch_size, seq_len, d_model)
    We impose headsize = d_model / num_heads. This is to make sure the FLOP cost
    is the same as a single headed attention with headsize = d_model.
    
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert self.d_model % self.n_heads == 0
        self.headsize = d_model // self.n_heads
        self.lq = Linear(self.d_model, self.d_model, nonlin=False)
        self.lk = Linear(self.d_model, self.d_model, nonlin=False)
        self.lv = Linear(self.d_model, self.d_model, nonlin=False)
        self.lout = Linear(self.d_model, self.d_model, nonlin=False)

    def __call__(self, x):
        q = self.lq(x)
        k = self.lk(x)
        v = self.lv(x)
        d_model = x.data.shape[-1]
        q = q.reshape(
            *q.data.shape[:-1], self.n_heads, self.headsize).transpose(-3, -2)
        k = k.reshape(
            *k.data.shape[:-1], self.n_heads, self.headsize).transpose(-3, -2)
        v = v.reshape(
            *v.data.shape[:-1], self.n_heads, self.headsize).transpose(-3, -2)
        out = masked_attention()(q, k, v)
        out = out.transpose(-3, -2)
        out = out.reshape(*out.data.shape[:-2], d_model)
        out = self.lout(out)
        return out
    
    def parameters(self):
        params = self.lq.parameters() + self.lk.parameters() \
                    + self.lv.parameters() +self.lout.parameters()
        return params
    
    def __repr__(self):
        return f'masked_multi_head_attention: d_model = {self.d_model}, \
                    n_heads = {self.n_heads}'

class Sequence(Module):

    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params
    
class Embedding(Module):
    """
    intput: Tnsr(*,) with entries in range(vocab_size)
    output: Tnsr(*, d_model)
    """

    def __init__(self, classes, d_emb):
        # language model: classes = vocab_size
        #                 d_emb = d_model
        self.classes = classes
        self.d_emb = d_emb
        self.w = Tnsr(np.random.randn(d_emb, classes) / np.sqrt(classes))

    def __call__(self, x):
        # We start by a one-hot encoding
        Y = np.arange(self.classes)
        X = np.expand_dims(x.data, axis=-1)
        one_hot = Tnsr(X == Y)
        # And apply a linear layer to it
        return one_hot.matmul(self.w.transpose(-1, -2))
    
    def parameters(self):
        return [self.w]
    
    def __repr__(self):
        return f'Embedding: classes = {self.classes}, d_emb = {self.d_emb}'
    

class DecoderBlock(Module):
    def __init__(self, n_heads, d_model):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.mmh = masked_multi_head_attention(self.d_model, self.n_heads)
        self.LN1 = LayerNorm(-1, self.d_model)
        self.LN2 = LayerNorm(-1, self.d_model)
        self.linear1 = Linear(self.d_model, 4*self.d_model, nonlin=True)
        self.linear2 = Linear(4*self.d_model, self.d_model, nonlin=False)
    
    def __call__(self, x):
        """
        x: Tnsr(batch_size, seq_len, d_model)
        """
        #masked multi head attention + add and norm
        x = self.LN1(self.mmh(x) + x)
        #Feed forward
        x = self.LN2(x + self.linear2(self.linear1(x)))
        return x
    
    def parameters(self):
        return self.mmh.parameters() + self.LN1.parameters() + \
                 self.LN2.parameters() + self.linear1.parameters() + \
                    self.linear2.parameters()
    
    def __repr__(self):
        return f'DecoderBlock: n_heads = {self.n_heads}, d_model = {self.d_model}'

    


class Transformer(Module):
    """
    Defines a deconding-only transformer
    """
    def __init__(self, N, n_heads, d_model, n_blocks, vocab_size):
        super().__init__()
        self.N = N
        self.n_heads = n_heads
        self.d_model = d_model 
        self.n_blocks = n_blocks
        self.embed = Embedding(vocab_size, d_model)
        self.pos_enc = pos_encoding(N, d_model)
        self.DecoderBlocks = Sequence(
            [DecoderBlock(n_heads, d_model) for _ in range(n_blocks)]
        )
        self.LN = LayerNorm(-1, d_model)
        self.linear_output = Linear(
            d_model, vocab_size, nonlin=False, bias=False)
        
    def __call__(self, x):
        """
        x: Tnsr(batch_size, seq_length)
        elements of x are integers in range(vocab_size)
        out: Tnsr(batch_size, seq_length, vocab_size)
        ouput = proba over vocabulary
        """
        
        seq_length = x.data.shape[-1]
        # Embedding
        x = self.embed(x) # output: Tnsr(batch_size, seq_length, d_model)
        # Positional encoding
        p = self.pos_enc(seq_length)
        x = x + p
        # Application n_blocks * (attention + feed forward)
        x = self.DecoderBlocks(x)
        # Layer norm
        x = self.LN(x)
        # Linear layer d_model -> vocab_size
        x = self.linear_output(x)
        return x
    
    def parameters(self):
        return self.embed.parameters() + self.DecoderBlocks.parameters() \
                + self.LN.parameters() + self.linear_output.parameters()
    
    def __repr__(self):
        return f'Encoding-only Transformer: N = {self.N}, \
              n_heads = {self.n_heads}, d_model = {self.d_model}, \
                  n_blocks = {self.n_blocks}, vocab_size = {self.vocab_size}'



