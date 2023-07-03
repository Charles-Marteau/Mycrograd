# Mycrograd

We build a toy version of pytorch which incorporates three key elements: 
- a toy version of the tensor class [my_engine.py](https://github.com/Charles-Marteau/Mycrograd/blob/main/my_engine.py)
- a toy version of torch.nn [my_nn.py](https://github.com/Charles-Marteau/Mycrograd/blob/main/my_nn.py)
- a toy version of torch.optim [my_optim.py](https://github.com/Charles-Marteau/Mycrograd/blob/main/my_optim.py)

Our version of the torch.tensor class is called 'Tnsr'. It allows for the creation of an abstract network
(thanks to the _children attribute) which represents the operations involving the Tnsr instantiations.


Exactly like the torch.tensor, a Tnsr possesses a gradient attribute and thanks to the creation of the
abstract network, automatic differentiation can be performed (this is captured by the .backward method).
