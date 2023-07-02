

# MYCROGRAD: a home-made autograd engine which defines a class called 'Tnsr'
# which mimics the tensor class of pytorch.


import numpy as np


class Tnsr():

    #---- Initialization ---------------------------------
    
    def __init__(self, data, _children=(), _op=''):
        # Data attribute is an array
        self.data = np.array(data, dtype=np.float64)
        # Attributes for network building
        self._children = _children
        self._prev = set(_children)
        self._op = _op
        # Attributes for gradient computations
        self._grad = np.zeros_like(self.data)
        self._backward = lambda :None

       
    def __repr__(self):
        return f'Tensor(\n{self.data}, op = {self._op})'

    
    #---- Operations ------------------------------------

    # We now list all the operations that one can do with Tnsrs

    # Schematically, operations are defined following the template:
    # 
    # def operation(self, other):
    #
    #   --- Define the output Tnsr with the output data ---
    #
    #   new_data = operation(self.data, other.data) 
    #   output = Tnsr(new_data, _children=(self, other), _op='operation') 
    #
    #   --- Define the backward attribute of the new Tnsr ---
    #   
    #   define _backward():
    #     self.grad = output.grad * d(operation)/d(self)
    #     other.grad = output.grad * d(operation)/d(other)
    #   
    #   output._backward = _backward
    #   return output
    
    
    def __getitem__(self, index):
        output = Tnsr(self.data[index], (self,))

        def _backward():
            grads = np.zeros_like(self.grad)
            grads[index] = output.grad
            self.grad += grads
        output._backward = _backward

        return output

    def __add__(self, other):
        other = other if isinstance(other, Tnsr) == True else Tnsr(other)
        sum = self.data + other.data
        output = Tnsr(sum, _children=(self, other), _op='+')
        def _backward():
            # We start by checking which tensor has the largest number of dimensions
            # And create two lists containing the dimensions of the ouput grad to be summed
            if self.data.ndim >= other.data.ndim:     
                delta = self.data.ndim - other.data.ndim
                dims_self = tuple(
                    np.where(np.array(self.data.shape[delta:]) == 1)[0] + delta
                    )
                dims_other = tuple(
                    [k for k in range(delta)]
                    ) \
                            + tuple(
                    np.where(np.array(other.data.shape) == 1)[0] + delta
                    )
            if self.data.ndim < other.data.ndim:
                delta = other.data.ndim - self.data.ndim
                dims_other = tuple(
                    np.where(np.array(other.data.shape[delta:]) == 1)[0] + delta
                    )
                dims_self = tuple(
                    [k for k in range(delta)]
                    ) \
                            + tuple(
                    np.where(np.array(self.data.shape) == 1)[0] + delta
                    )
            # The gradient of each tensor is defined to be the output one,
            # summed over the broadcasted dimensions
            self.grad += output.grad.sum(axis=dims_self, keepdims=True).reshape(self.data.shape)
            other.grad += output.grad.sum(axis=dims_other, keepdims=True).reshape(other.data.shape)
        output._backward = _backward
        return output   
    
    def __mul__(self, other):
        other = other if isinstance(other, Tnsr) == True else Tnsr(other)
        prod = self.data * other.data
        output = Tnsr(prod, _children=(self, other), _op='*')
        def _backward():
            # We start by checking which tensor has the largest number of dimensions
            # And create two lists containing the ouput grad dims to be summed
            if self.data.ndim >= other.data.ndim:     
                delta = self.data.ndim - other.data.ndim
                dims_self = tuple(
                    np.where(np.array(self.data.shape[delta:]) == 1)[0] + delta
                    )
                dims_other = tuple(
                    [k for k in range(delta)]
                    ) \
                            + tuple(
                    np.where(np.array(other.data.shape) == 1)[0] + delta
                    )
            if self.data.ndim < other.data.ndim:
                delta = other.data.ndim - self.data.ndim
                dims_other = tuple(
                    np.where(np.array(other.data.shape[delta:]) == 1)[0] + delta
                    )
                dims_self = tuple(
                    [k for k in range(delta)]
                    ) \
                            + tuple(
                    np.where(np.array(self.data.shape) == 1)[0] + delta
                    )
            # The gradient of each tensor is defined to be the output one,
            # summed over the broadcasted dimensions
            self.grad += (
                output.grad * other.data
                ).sum(axis=dims_self, keepdims=True).reshape(self.data.shape)
            other.grad += (
                output.grad * self.data
                ).sum(axis=dims_other, keepdims=True).reshape(other.data.shape)
        output._backward = _backward
        return output  
    
    def __neg__(self):
        return self * np.float64(-1)
    
    def __sub__(self, other):
        return self + (-other)
    
    def matmul(self, other):
        other = other if isinstance(other, Tnsr) == True else Tnsr(other)
        matmul = self.data.__matmul__(other.data)
        output = Tnsr(matmul, _children=(self, other), _op=f'matmul')
        def _backward():
            # There are three cases in the definition of mp.matmul
            if len(self.data.shape) == 1 and len(other.data.shape) == 1:
                # dot product of vectors
                self.grad += other.data * output.grad
                other.grad += self.data * output.grad
            elif len(self.data.shape) > 1 and len(other.data.shape) == 1:
                # vector multiplying matrix from left
                self.grad += np.tensordot(output.grad, other.data, axes=0)
                other.grad += np.tensordot(output.grad, self.data, axes=len(output.grad.shape))
            elif len(self.data.shape) == 1 and len(other.data.shape) > 1:
                other.grad += np.tensordot(output.grad, self.data, axes=0).swapaxes(-1, -2)
                self.grad += np.tensordot(
                    output.grad, other.data.swapaxes(-1, -2), axes=len(output.grad.shape)
                    )
            else:
                # matrix multiplication
                # We start by checking which tensor has the largest number of dimensions
                # And create two lists containing the dimensions of the ouput grad to be summed
                if self.data.ndim >= other.data.ndim:     
                    delta = self.data.ndim - other.data.ndim
                    dims_self = tuple(
                        np.where(np.array(self.data.shape[delta:]) == 1)[0] + delta
                        )
                    dims_other = tuple(
                        [k for k in range(delta)]
                        ) + tuple(
                        np.where(np.array(other.data.shape) == 1)[0] + delta
                        )
                if self.data.ndim < other.data.ndim:
                    delta = other.data.ndim - self.data.ndim
                    dims_other = tuple(
                        np.where(np.array(other.data.shape[delta:]) == 1)[0] + delta
                        )
                    dims_self = tuple(
                        [k for k in range(delta)]
                        ) + tuple(
                        np.where(np.array(self.data.shape) == 1)[0] + delta
                        )
                # The gradient of each tensor is defined to be the appropriate matmul 
                # with the ouput gradient, summed over the broadcasted dimensions
                self.grad += output.grad.__matmul__(other.data.swapaxes(-1, -2)).sum(
                        axis=dims_self, keepdims=True).reshape(self.data.shape)
                other.grad += self.data.swapaxes(-1, -2).__matmul__(output.grad).sum(
                    axis=dims_other, keepdims=True).reshape(other.data.shape)
        output._backward = _backward
        return output

    def __pow__(self, other):
        other = other if isinstance(other, Tnsr) == True else Tnsr(other)
        pow = self.data ** other.data
        output = Tnsr(pow, _children=(self, other), _op='**')
        def _backward():
            # We start by checking which tensor has the largest number of dimensions
            # And create two lists containing the ouput grad dims to be summed
            if self.data.ndim >= other.data.ndim:     
                delta = self.data.ndim - other.data.ndim
                dims_self = tuple(
                    np.where(np.array(self.data.shape[delta:]) == 1)[0] + delta
                    )
                dims_other = tuple(
                    [k for k in range(delta)]
                    ) \
                            + tuple(
                    np.where(np.array(other.data.shape) == 1)[0] + delta
                    )
            if self.data.ndim < other.data.ndim:
                delta = other.data.ndim - self.data.ndim
                dims_other = tuple(
                    np.where(np.array(other.data.shape[delta:]) == 1)[0] + delta
                    )
                dims_self = tuple(
                    [k for k in range(delta)]
                    ) \
                            + tuple(
                    np.where(np.array(self.data.shape) == 1)[0] + delta
                    )
            # d(x^y)/dx = y * x^(y-1),
            # summed over the broadcasted dimensions
            self.grad += (
                output.grad * other.data * self.data ** (other.data - 1)
                ).sum(axis=dims_self, keepdims=True).reshape(self.data.shape)
            # d(x^y)/dy = logx x^y,
            # summed over the broadcasted dimensions
            other.grad += (
                output.grad * np.log(self.data) * (self.data ** other.data)
                ).sum(axis=dims_other, keepdims=True).reshape(other.data.shape)
        output._backward = _backward
        return output  
    
    def __truediv__(self, other):
        return self * (other ** np.float64(-1))
    

    def relu(self):
        output = Tnsr(np.maximum(self.data, 0.), _children=(self,), _op='relu')
        def _backward():
            self.grad += (output.data > 0) * output.grad
        output._backward = _backward
        return output

    def max(self, axis=None, keepdims=False):
        output = Tnsr(
            self.data.max(axis=axis, keepdims=keepdims),
              _children=(self,), _op=f'max(axis={axis}, keepdims={keepdims})'
            )
        def _backward():
            if axis==None: ax = tuple(range(len(self.data.shape)))
            else: ax = axis
            if keepdims == False: 
                # Restaure the dimensions that were maxed
                output_data = np.expand_dims(output.data, axis=ax) 
                output_grad = np.expand_dims(output.grad, axis=ax)
            else:
                output_data = output.data
                output_grad = output.grad
            mask = output_data == self.data # Equality is broadcasted over maxed axis
            count = mask.sum(axis=ax, keepdims=True) # Counts duplicate maximum elements
            self.grad += output_grad * mask / count # Multiplication is broadcasted too
        output._backward = _backward
        return output
    
    def reshape(self, *shape):
        output = Tnsr(self.data.reshape(*shape), _children=(self,), _op="reshape")
        def _backward():
            self.grad += output.grad.reshape(*self.data.shape)
        output._backward = _backward
        return output
    
    def transpose(self, dim0, dim1):
        output = Tnsr(np.swapaxes(self.data, dim0, dim1), _children=(self,), _op="transpose")
        def _backward():
            self.grad += np.swapaxes(output.grad, dim0, dim1)
        output._backward = _backward
        return output
    
    def sum(self, axis=None, keepdims=False):
        output = Tnsr(
            self.data.sum(axis=axis, keepdims=keepdims),
              _children=(self,), _op=f'sum(axis={axis}, keepdims={keepdims})'
            )
        def _backward():
            if axis==None: ax = tuple(range(len(self.data.shape)))
            else: ax = axis
            if keepdims == False: 
                # Restaure the dimensions that were summed
                output_grad = np.expand_dims(output.grad, axis=ax)
            else: output_grad = output.grad
            self.grad += output_grad # Addition is broadcasted to match self.grad.shape
        output._backward = _backward
        return output
    
    def mean(self, axis=None, keepdims=False):
        if axis==None: ax = tuple(range(len(self.data.shape)))
        elif type(axis) == int: ax = (axis,)
        else: ax = axis
        
        count = np.array([self.data.shape[k] for k in ax]).prod()
        output = Tnsr(
            self.data.sum(axis=ax, keepdims=keepdims) / count,
              _children=(self,), _op=f'mean(axis={ax}, keepdims={keepdims})'
            )
        def _backward():
            if keepdims == False: 
                # Restaure the dimensions that were averaged
                output_grad = np.expand_dims(output.grad, axis=ax)
            else: output_grad = output.grad
            self.grad += output_grad / count # Addition is broadcasted to match self.grad.shape
        output._backward = _backward
        return output
    
    def exp(self):
        output = Tnsr(np.exp(self.data), _children=(self,), _op='exp')
        def _backward():
            self.grad += output.grad * np.exp(self.data)
        output._backward =_backward
        return output
    
    def log(self):
        output = Tnsr(np.log(self.data), _children=(self,), _op='log')
        def _backward():
            self.grad += output.grad / self.data
        output._backward =_backward
        return output
    
    def sqrt(self):
        output = Tnsr(np.sqrt(self.data), _children=(self,), _op='sqrt')
        def _backward():
            self.grad += output.grad / (2 * np.sqrt(self.data))
        output._backward =_backward
        return output
    
    def mask(self):
        # Masks the upper triangular part of a matrix (above diagonal)
        mask = np.triu(np.ones(self.data.shape, dtype=bool), k=1)
        output_data = self.data.copy()
        output_data[mask] = - 1e9
        output = Tnsr(output_data, _children=(self,), _op='mask')
        def _backward():
            self_grad = output.grad.copy()
            self_grad[mask] = 0
            self.grad += self_grad
        output._backward = _backward
        return output
    
    def cat(self, other, axis=0):
        # Works a bit differently than torch.cat.
        # Implicit ordering torch.cat(self, other) by self.cat(other)
        other = other if isinstance(other, Tnsr) else Tnsr(other)
        output = Tnsr(
            np.concatenate((self.data, other.data), axis=axis),
              _children=(self,other), _op='cat')

        def _backward():
            self.grad += output.grad.take(indices=range(self.shape[axis]), axis=axis)
            other.grad += output.grad.take(
                indices=range(self.shape[axis], output.shape[axis]), axis=axis
                )
        output._backward = _backward
        return output
 
    
    #---- Backward ----------------------------------------

    # We now define the method Tnsr.backward(), which starts from
    # the output tensor and populates the gradients of all the nodes of the network
    # by applying successive _backward()s to a topologically ordered version of the
    # network graph.
    
   
    def backward(self):
        # Topological ordering of the graph where self is the last one
        topo = []
        visited = set() # To avoid double counting due to loops

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)  

        build_topo(self)
        self.grad = np.ones_like(self.data) # dself/dself = 1
        for v in reversed(topo):
            v._backward()
    
    #---- Consistency conditions ----------------------------

    # Verification that consistency conditions between data and grad are satisfied.

    @property # Function automatically called when calling self.grad
    def grad(self):
        return self._grad  
    
    @grad.setter # Function automatically called when trying to assign a value to self.grad
    def grad(self, new_grad):
        if type(new_grad) != type(self.data):
            raise ValueError(
                f"data type ({type(self.data)}) is different \
                    from input grad type ({type(new_grad)})")
        elif new_grad.dtype != self.data.dtype:
            raise ValueError(
                f"data dtype ({self.data.dtype}) is different \
                    from input grad dtype ({new_grad.dtype})")
        elif new_grad.shape != self.data.shape:
            raise ValueError(
                f"data shape ({self.data.shape}) is different \
                    from input grad type ({new_grad.shape})")
        else: self._grad = new_grad