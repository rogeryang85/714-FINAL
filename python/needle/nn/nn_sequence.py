"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        one = init.constant(*x.shape, c=1.0, device=x.device, dtype=x.dtype, requires_grad=True)
        return one / (one + ops.exp(-x))
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        k = 1 / np.sqrt(hidden_size)
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity

        # Initialize weights
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))

        # Initialize biases if bias is True
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        if h is None:
            h = init.constant(batch_size, self.hidden_size, c=0.0, device=X.device, dtype=X.dtype)

        # Compute the next hidden state
        linear_combination = ops.matmul(X, self.W_ih) + ops.matmul(h, self.W_hh)
        # print("linear_combination", linear_combination.shape)
        # print("bias_ih", self.bias_ih.shape, ";hidden_size ", self.hidden_size, ";batch_size ", batch_size)
        if self.bias_ih is not None:
            bias_ih = ops.reshape(self.bias_ih, (1, self.hidden_size))
            linear_combination += ops.broadcast_to(bias_ih, (batch_size, self.hidden_size))
        if self.bias_hh is not None:
            bias_hh = ops.reshape(self.bias_hh, (1, self.hidden_size))
            linear_combination += ops.broadcast_to(bias_hh, (batch_size, self.hidden_size))

        # Apply the nonlinearity
        if self.nonlinearity == 'tanh':
            h_next = ops.tanh(linear_combination)
        elif self.nonlinearity == 'relu':
            h_next = ops.relu(linear_combination)
        else:
            raise ValueError(f"Invalid nonlinearity {self.nonlinearity}")

        return h_next
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Initialize RNN cells for each layer
        self.rnn_cells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            rnn_cell = RNNCell(layer_input_size, hidden_size, bias=bias, nonlinearity=nonlinearity, device=device, dtype=dtype)
            self.rnn_cells.append(rnn_cell)

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs:
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        seq_len, batch_size, _ = X.shape
        device = X.device
        dtype = X.dtype

        # Split h0 if provided, otherwise initialize it
        if h0 is None:
            h0 = [init.constant(batch_size, self.hidden_size, c=0.0, device=device, dtype=dtype) for _ in range(self.num_layers)]
        else:
            h0 = ops.split(h0, axis=0)  # Split into tensors for each layer

        # Split the input sequence along the time axis
        X_slices = ops.split(X, axis=0)

        # Prepare to store outputs at each time step
        output_seq = []
        h_n = h0

        # Iterate over each time step
        for t in range(seq_len):
            x_t = X_slices[t]
            h_t = []

            # Iterate over each layer
            for layer in range(self.num_layers):
                h_prev = h_n[layer]
                rnn_cell = self.rnn_cells[layer]
                h_t_next = rnn_cell(x_t, h_prev)
                h_t.append(h_t_next)
                x_t = h_t_next  # Output of this layer becomes input to the next

            h_n = h_t
            output_seq.append(h_t[-1])  # Append output of the last layer for this time step

        # Stack outputs and final hidden states
        output = ops.stack(output_seq, axis=0)
        h_n = ops.stack(h_n, axis=0)

        return output, h_n





class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights.

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size.
        """
        super().__init__()
        k = 1 / np.sqrt(hidden_size)
        self.hidden_size = hidden_size
        self.sigmoid = Sigmoid()

        # Initialize weights
        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))

        # Initialize biases if applicable
        if bias:
            self.bias_ih = Parameter(init.rand(4 * hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(4 * hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features.
        h, tuple of (h0, c0), with:
            h0 of shape (batch, hidden_size): Initial hidden state for each element in the batch.
            c0 of shape (batch, hidden_size): Initial cell state for each element in the batch.

        Outputs: (h', c')
        h' of shape (batch, hidden_size): Next hidden state for each element in the batch.
        c' of shape (batch, hidden_size): Next cell state for each element in the batch.
        """
        batch_size = X.shape[0]

        # Initialize h0 and c0 if not provided
        if h is None:
            h = (
                init.constant(batch_size, self.hidden_size, c=0.0, device=X.device, dtype=X.dtype),
                init.constant(batch_size, self.hidden_size, c=0.0, device=X.device, dtype=X.dtype),
            )

        h0, c0 = h

        # Linear transformations
        gates = ops.matmul(X, self.W_ih) + ops.matmul(h0, self.W_hh)

        # Add biases if present
        if self.bias_ih is not None:
            gates += ops.broadcast_to(ops.reshape(self.bias_ih, (1, 4 * self.hidden_size)), gates.shape)
        if self.bias_hh is not None:
            gates += ops.broadcast_to(ops.reshape(self.bias_hh, (1, 4 * self.hidden_size)), gates.shape)

        # Reshape gates for splitting
        gates = ops.reshape(gates, (batch_size, 4, self.hidden_size))

        # Split gates into input, forget, cell, and output gates
        i, f, g, o = ops.split(gates, axis=1)

        # Apply activation functions
        i = self.sigmoid(ops.reshape(i, (batch_size, self.hidden_size)))
        f = self.sigmoid(ops.reshape(f, (batch_size, self.hidden_size)))
        g = ops.tanh(ops.reshape(g, (batch_size, self.hidden_size)))
        o = self.sigmoid(ops.reshape(o, (batch_size, self.hidden_size)))

        # Compute next cell state and hidden state
        c_next = f * c0 + i * g
        h_next = o * ops.tanh(c_next)

        return h_next, c_next







class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Initialize LSTM cells for each layer
        self.lstm_cells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            lstm_cell = LSTMCell(layer_input_size, hidden_size, bias=bias, device=device, dtype=dtype)
            self.lstm_cells.append(lstm_cell)

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial hidden state.
            c_0 of shape (num_layers, bs, hidden_size) containing the initial cell state.
            Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features.
        tuple of (h_n, c_n) with:
            h_n: Final hidden state for each element in the batch.
            c_n: Final cell state for each element in the batch.
        """
        seq_len, batch_size, _ = X.shape
        device = X.device
        dtype = X.dtype

        # Initialize hidden states if not provided
        if h is None:
            h = (
                init.constant(self.num_layers, batch_size, self.hidden_size, c=0.0, device=device, dtype=dtype),
                init.constant(self.num_layers, batch_size, self.hidden_size, c=0.0, device=device, dtype=dtype),
            )
        h0, c0 = h

        # Split h0 and c0 into tensors for each layer
        h_n = ops.split(h0, axis=0)
        c_n = ops.split(c0, axis=0)

        # Split the input sequence along the time axis
        X_slices = ops.split(X, axis=0)

        output_seq = []

        # Iterate through the sequence
        for t in range(seq_len):
            x_t = X_slices[t]
            h_t, c_t = [], []

            # Iterate through each layer
            for layer in range(self.num_layers):
                h_prev = h_n[layer]
                c_prev = c_n[layer]
                lstm_cell = self.lstm_cells[layer]
                h_next, c_next = lstm_cell(x_t, (h_prev, c_prev))
                h_t.append(h_next)
                c_t.append(c_next)
                x_t = h_next  # The output of this layer becomes the input for the next

            h_n = h_t
            c_n = c_t
            output_seq.append(h_t[-1])  # Output of the last layer at time t

        # Stack outputs and hidden states for final output
        output = ops.stack(output_seq, axis=0)
        h_n = ops.stack(h_n, axis=0)
        c_n = ops.stack(c_n, axis=0)

        return output, (h_n, c_n)





class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Initialize embedding weights from N(0,1)
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, mean=0.0, std=1.0, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        # Create a list to hold embedding results for each time step
        embeddings = []

        # Split along the seq_len dimension
        x_slices = ops.split(x, axis=0)

        # Process each time step independently
        for x_t in x_slices:
            # Flatten batch dimension for this time step
            bs, = x_t.shape
            x_flat = ops.reshape(x_t, (bs,))
            # Generate one-hot vectors for batch indices
            one_hot = init.one_hot(self.num_embeddings, x_flat, device=x.device, dtype=x.dtype)
            # Perform matrix multiplication with weight
            embedding_t = ops.matmul(one_hot, self.weight)
            # Append to the list of embeddings
            embeddings.append(embedding_t)

        # Stack embeddings along seq_len dimension
        output = ops.stack(embeddings, axis=0)
        return output
        ### END YOUR SOLUTION