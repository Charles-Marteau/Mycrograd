# Mycrograd

Using only the numpy library, we build a toy version of pytorch which incorporates three key elements: 
- a toy version of the tensor class: [my_engine.py](https://github.com/Charles-Marteau/Mycrograd/blob/main/my_engine.py)
- a toy version of torch.nn: [my_nn.py](https://github.com/Charles-Marteau/Mycrograd/blob/main/my_nn.py)
- a toy version of torch.optim: [my_optim.py](https://github.com/Charles-Marteau/Mycrograd/blob/main/my_optim.py)

Armed with this, we conduct experiments to test the functionality of Mycrograd and ensure proper interaction among its various components, this is done in [experiment.ipynb](https://github.com/Charles-Marteau/Mycrograd/blob/main/experiment.ipynb).

### my_engine.py

Our version of the torch.tensor class is called 'Tnsr'. It allows for the creation of an abstract network
(thanks to the _children attribute) which represents the operations involving the Tnsr instantiations.

Exactly like the torch.tensor, a Tnsr possesses a gradient attribute and thanks to the creation of the
abstract network, automatic differentiation can be performed (this is captured by the .backward method).

Most usual operations (multiplication, addition, power, matmul, indexing ...) are covered.

### my_nn.py

We define various layers necessary to the construction of a feed forward neural network and a transformer. These layers are
- Linear Layer
- Feed forward (successive linear layers with ReLU activations in between)
- Softmax
- Cross entropy
- Positional encoding (to be added to the embedding in the transformer)
- Layer norm
- Masked Attention 
- Masked multi_head attention
- Sequence
- Embedding
- Decoder block
- Transformer

### my_optim.py 

We define two optimizers: stochastic gradient descent (SGD) and Adam. These optimizers interact with the layer parameters of my_nn by updating their values in a direction that is dictated by one of these optimization algorithm.

### experiment.ipynb

In this notebook we test our layers, our automatic differentiation and our optimizers by putting everything together and training neural networks.
We define a feed forward neural network and train it on a random dataset. Then we train a transformer to do addition of two digits number. We conclude with a coparison with torch.

