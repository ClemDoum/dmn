import theano
import theano.tensor as T
from lasagne import init, nonlinearities
from lasagne.layers import MergeLayer, Gate
from theano.gradient import np


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = T.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = T.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


class GatingLayer(object):
    def __init__(self,
                 incoming,
                 representation_dim,
                 mask_input=None,
                 learn_init=True,
                 gradient_steps=-1,
                 **kwargs):
        # This layer inherits from a MergeLayer, because it can have two
        # inputs - the layer input, and the mask.  We will just provide the
        # layer input as incomings, unless a mask input was provided.
        incomings = [incoming]
        if mask_input is not None:
            incomings.append(mask_input)

        self.representation_dim = representation_dim
        self.W_b = theano.shared(
            init.Normal(0.1).sample((representation_dim, representation_dim)),
            name="W_b")
        self.W_1 = theano.shared(
            init.Normal(0.1).sample((representation_dim, representation_dim)),
            name="W_1")
        self.W_2 = theano.shared(
            init.Normal(0.1).sample((representation_dim, representation_dim)),
            name="W_2")
        self.b_1 = theano.shared(
            init.Constant(0.).sample((representation_dim,)), name="b_1")
        self.b_2 = theano.shared(
            init.Constant(0.).sample(representation_dim, ), name="b_2")

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        return input_shape[0], input_shape[1], self.re

    def get_output_for(self, inputs):
        # Retrieve the layer input
        fact = inputs[0]
        memory = inputs[1]
        question = inputs[2]

        # Output must be of shape
        # num_batches * dim * concatenation_length

        # Compute z
        to_concatenate = list()

        def add_axis_and_extend(to_concatenate, tensor_list, last_dim=1):
            for t in tensor_list:
                to_concatenate.append(t.reshape((t.shape[0], t.shape[1],
                                                 last_dim)))

        def reshape(t, last_dim=1):
            return t.reshape((t.shape[0], t.shape[1], last_dim))

        add_axis_and_extend(to_concatenate, [fact, memory, question])
        add_axis_and_extend(to_concatenate, [fact * question, fact * memory])
        add_axis_and_extend(to_concatenate,
                            [T.abs_(fact - question), T.abs_(fact - memory)])
        to_concatenate.extend([T.dot(fact.T, T.dot(question, self.W_b)),
                               T.dot(fact.T, T.dot(memory, self.W_b))])

        # return reshape(fact.T)
        # return reshape(T.dot(question, self.W_b))
        return T.dot(question, self.W_b)
        return T.dot(reshape(fact.T), T.dot(question, self.W_b))

        return concatenate(to_concatenate, axis=2)
        print len(to_concatenate)
        return concatenate(to_concatenate, axis=1)

        z = concatenate(to_concatenate, axis=2)

        # Compute the gates
        gates = T.dot(T.tanh(T.dot(z, self.W_1) + self.b_1),
                      self.W_2) + self.b_2
        gates = T.nnet.sigmoid(gates)
        return gates


class ModifiedGruLayer(object):
    def __init__(self,
                 incoming,
                 representation_dim,
                 mask_input=None,
                 learn_init=True,
                 gradient_steps=-1,
                 **kwargs):
        # This layer inherits from a MergeLayer, because it can have two
        # inputs - the layer input, and the mask.  We will just provide the
        # layer input as incomings, unless a mask input was provided.
        incomings = [incoming]
        if mask_input is not None:
            incomings.append(mask_input)

        self.representation_dim = representation_dim
        self.W_b = theano.shared(
            init.Normal(0.1).sample((representation_dim, representation_dim)),
            name="W_b")
        self.W_1 = theano.shared(
            init.Normal(0.1).sample((representation_dim, representation_dim)),
            name="W_1")
        self.W_2 = theano.shared(
            init.Normal(0.1).sample((representation_dim, representation_dim)),
            name="W_2")
        self.b_1 = theano.shared(
            init.Constant(0.).sample((representation_dim,)), name="b_1")
        self.b_2 = theano.shared(
            init.Constant(0.).sample(representation_dim, ), name="b_2")


class EpisodicMemoryLayer(MergeLayer):
    r"""A long short-term memory (LSTM) layer.
    Includes optional "peephole connections" and a forget gate.  Based on the
    definition in [1]_, which is the current common definition.  The output is
    computed by
    .. math ::
        i_t &= \sigma_i(W_{xi}x_t + W_{hi}h_{t-1}
               + w_{ci}\odot c_{t-1} + b_i)\\
        f_t &= \sigma_f(W_{xf}x_t + W_{hf}h_{t-1}
               + w_{cf}\odot c_{t-1} + b_f)\\
        c_t &= f_t \odot c_{t - 1}
               + i_t\sigma_c(W_{xc}x_t + W_{hc} h_{t-1} + b_c)\\
        o_t &= \sigma_o(W_{xo}x_t + W_{ho}h_{t-1} + w_{co}\odot c_t + b_o)\\
        h_t &= o_t \odot \sigma_h(c_t)
    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden/cell units in the layer.
    W_in_to_ingate : Theano shared variable, numpy array or callable
        Initializer for input-to-input gate weight matrix (:math:`W_{xi}`).
    W_hid_to_ingate : Theano shared variable, numpy array or callable
        Initializer for hidden-to-input gate weight matrix (:math:`W_{hi}`).
    W_cell_to_ingate : Theano shared variable, numpy array or callable
        Initializer for cell-to-input gate weight vector (:math:`w_{ci}`).
    b_ingate : Theano shared variable, numpy array or callable
        Initializer for input gate bias vector (:math:`b_i`).
    nonlinearity_ingate : callable or None
        The nonlinearity that is applied to the input gate activation
        (:math:`\sigma_i`). If None is provided, no nonlinearity will be
        applied.
    W_in_to_forgetgate : Theano shared variable, numpy array or callable
        Initializer for input-to-forget gate weight matrix (:math:`W_{xf}`).
    W_hid_to_forgetgate : Theano shared variable, numpy array or callable
        Initializer for hidden-to-forget gate weight matrix (:math:`W_{hf}`).
    W_cell_to_forgetgate : Theano shared variable, numpy array or callable
        Initializer for cell-to-forget gate weight vector (:math:`w_{cf}`).
    b_forgetgate : Theano shared variable, numpy array or callable
        Initializer for forget gate bias vector (:math:`b_f`).
    nonlinearity_forgetgate : callable or None
        The nonlinearity that is applied to the forget gate activation
        (:math:`\sigma_f`). If None is provided, no nonlinearity will be
        applied.
    W_in_to_cell : Theano shared variable, numpy array or callable
        Initializer for input-to-cell weight matrix (:math:`W_{ic}`).
    W_hid_to_cell : Theano shared variable, numpy array or callable
        Initializer for hidden-to-cell weight matrix (:math:`W_{hc}`).
    b_cell : Theano shared variable, numpy array or callable
        Initializer for cell bias vector (:math:`b_c`).
    nonlinearity_cell : callable or None
        The nonlinearity that is applied to the cell activation
        (;math:`\sigma_c`). If None is provided, no nonlinearity will be
        applied.
    W_in_to_outgate : Theano shared variable, numpy array or callable
        Initializer for input-to-output gate weight matrix (:math:`W_{io}`).
    W_hid_to_outgate : Theano shared variable, numpy array or callable
        Initializer for hidden-to-output gate weight matrix (:math:`W_{ho}`).
    W_cell_to_outgate : Theano shared variable, numpy array or callable
        Initializer for cell-to-output gate weight vector (:math:`w_{co}`).
    b_outgate : Theano shared variable, numpy array or callable
        Initializer for hidden-to-input gate weight matrix (:math:`b_o`).
    nonlinearity_outgate : callable or None
        The nonlinearity that is applied to the output gate activation
        (:math:`\sigma_o`). If None is provided, no nonlinearity will be
        applied.
    nonlinearity_out : callable or None
        The nonlinearity that is applied to the output (:math:`\sigma_h`). If
        None is provided, no nonlinearity will be applied.
    cell_init : callable, np.ndarray, theano.shared or TensorVariable
        Passing in a TensorVariable allows the user to specify
        the value of `cell_init` (:math:`c_0`). In this mode `learn_init` is
        ignored for the cell state.
    hid_init : callable, np.ndarray, theano.shared or TensorVariable
        Passing in a TensorVariable allows the user to specify
        the value of `hid_init` (:math:`h_0`). In this mode `learn_init` is
        ignored for the hidden state.
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned. If `hid_init` or
        `cell_init` are TensorVariables then the TensorVariable is used and
        `learn_init` is ignored for that initial state.
    peepholes : bool
        If True, the LSTM uses peephole connections.
        When False, `W_cell_to_ingate`, `W_cell_to_forgetgate` and
        `W_cell_to_outgate` are ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping: False or float
        If a float is provided, the gradient messages are clipped during the
        backward pass.  If False, the gradients will not be clipped.  See [1]_
        (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is true then the `gradient_steps`
        setting is ignored.
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    References
    ----------
    .. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    """

    def __init__(self,
                 incoming,
                 num_units,
                 n_decodesteps,
                 resetgate=Gate(W_cell=None),
                 updategate=Gate(W_cell=None),
                 hidden_update=Gate(W_cell=None,
                                    nonlinearity=nonlinearities.tanh),
                 hid_init=init.Constant(0.),
                 grad_clipping=False,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 **kwargs):
        # Initialize parent layer
        # This layer inherits from a MergeLayer, because it can have two
        # inputs - the layer input, and the mask.  We will just provide the
        # layer input as incomings, unless a mask input was provided.
        incomings = [incoming]
        if mask_input is not None:
            incomings.append(mask_input)

        super(EpisodicMemoryLayer, self).__init__(incomings, **kwargs)
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.precompute_input = precompute_input

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in all parameters from gates
        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')

        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
            hidden_update, 'hidden_update')

    # At each call to scan, input_n will be (n_time_steps, 3*num_units).
    # We define a slicing function that extract the input to each GRU gate
    def slice_w(self, x, n):
        return x[:, n * self.num_units:(n + 1) * self.num_units]

    def gru_step(self, input_n, hid_previous, W_hid_stacked, W_in_stacked,
                 b_stacked):
        # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
        hid_input = T.dot(hid_previous, W_hid_stacked)

        if self.grad_clipping is not False:
            input_n = theano.gradient.grad_clip(
                input_n, -self.grad_clipping, self.grad_clipping)
            hid_input = theano.gradient.grad_clip(
                hid_input, -self.grad_clipping, self.grad_clipping)

        if not self.precompute_input:
            # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
            input_n = T.dot(input_n, W_in_stacked) + b_stacked

        # Reset and update gates
        resetgate = self.slice_w(hid_input, 0) + self.slice_w(input_n, 0)
        updategate = self.slice_w(hid_input, 1) + self.slice_w(input_n, 1)
        resetgate = self.nonlinearity_resetgate(resetgate)
        updategate = self.nonlinearity_updategate(updategate)

        # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
        hidden_update_in = self.slice_w(input_n, 2)
        hidden_update_hid = self.slice_w(hid_input, 2)
        hidden_update = hidden_update_in + resetgate * hidden_update_hid
        if self.grad_clipping is not False:
            hidden_update = theano.gradient.grad_clip(
                hidden_update, -self.grad_clipping, self.grad_clipping)
        hidden_update = self.nonlinearity_hid(hidden_update)

        # Compute (1 - u_t)h_{t - 1} + u_t c_t
        hid = (1 - updategate) * hid_previous + updategate * hidden_update
        return hid

    def masked_gru_step(self, input_n, mask_n, hid_previous, W_hid_stacked,
                        W_in_stacked, b_stacked):

        hid = self.gru_step(input_n, hid_previous, W_hid_stacked, W_in_stacked,
                            b_stacked)

        # Skip over any input with mask 0 by copying the previous
        # hidden state; proceed normally for any input with mask 1.
        not_mask = 1 - mask_n
        hid = hid * mask_n + hid_previous * not_mask

        return hid

    def inner_gru_step(self, input_n, hid_previous, attention_gate,
                       W_hid_stacked, W_in_stacked, b_stacked):
        hid = self.gru_step(input_n, hid_previous, W_hid_stacked,
                            W_in_stacked, b_stacked)
        hid = attention_gate * hid + (1 - attention_gate) * hid_previous
        return hid

    def masked_inner_gru_step(self, input_n, mask_n, hid_previous,
                              attention_gate, W_hid_stacked, W_in_stacked,
                              b_stacked):
        hid = self.inner_gru_step(input_n, hid_previous, attention_gate,
                                  W_hid_stacked, W_in_stacked, b_stacked)
        # Skip over any input with mask 0 by copying the previous
        # hidden state; proceed normally for any input with mask 1.
        # return hid
        not_mask = 1 - mask_n
        hid = hid * mask_n + hid_previous * not_mask
        return hid

    def episode(self, attentions, facts, questions, Wf, Uf, bf, dim,
                mask=None):
        if self.softmax_attention:
            NotImplementedError
        else:
            fact_dot_Wf_plus_bf = T.dot(facts, Wf) + bf
            seqs = [attentions, fact_dot_Wf_plus_bf]
            init_states = [questions]
            non_seqs = [Uf, dim]
            episodes, updates = theano.scan(compute_modified_gru_state,
                                            sequences=seqs,
                                            outputs_info=init_states,
                                            non_sequences=non_seqs,
                                            name="gru_layer",
                                            strict=True)
            return episodes

    def attention_gate(self, inputs):
        # Retrieve the layer input
        fact = inputs[0]
        memory = inputs[1]
        question = inputs[2]

        # Output must be of shape
        # num_batches * dim * concatenation_length

        # Compute z
        to_concatenate = list()

        def add_axis_and_extend(to_concatenate, tensor_list, last_dim=1):
            for t in tensor_list:
                to_concatenate.append(t.reshape((t.shape[0], t.shape[1],
                                                 last_dim)))

        def reshape(t, last_dim=1):
            return t.reshape((t.shape[0], t.shape[1], last_dim))

        add_axis_and_extend(to_concatenate, [fact, memory, question])
        add_axis_and_extend(to_concatenate,
                            [fact * question, fact * memory])
        add_axis_and_extend(to_concatenate,
                            [T.abs_(fact - question),
                             T.abs_(fact - memory)])
        to_concatenate.extend([T.dot(fact.T, T.dot(question, self.W_b)),
                               T.dot(fact.T, T.dot(memory, self.W_b))])

        # return reshape(fact.T)
        # return reshape(T.dot(question, self.W_b))
        return T.dot(question, self.W_b)
        return T.dot(reshape(fact.T), T.dot(question, self.W_b))

        return concatenate(to_concatenate, axis=2)
        print len(to_concatenate)
        return concatenate(to_concatenate, axis=1)

        z = concatenate(to_concatenate, axis=2)

        # Compute the gates
        gates = T.dot(T.tanh(T.dot(z, self.W_1) + self.b_1),
                      self.W_2) + self.b_2
        gates = T.nnet.sigmoid(gates)
        return gates

    def get_episodic_memory(self, facts, questions, nb_passes, params):
        def compute_memory(f, q, m_1, U, W1, b1, W2, b2, Wb, Wf, Uf, bf,
                           Wm,
                           bm, dim):
            # 1. Compute the attention from the fact, memory and question
            a = self.attention_gates(f, q, m_1, W1, b1, W2, b2, Wb)
            # 2. Compute the episode from the attention and the facts
            e = self.episodes(a, f, q, Wf, Uf, bf, dim)
            # 3. Compute the new memory from the episode
            e_dot_Wm_plus_bm = T.dot(e, Wm) + bm
            m = gru_step(e_dot_Wm_plus_bm, m_1, U, dim)
            return m

        seqs = [facts]
        init_states = [T.alloc(questions)]  # Initialize it to the question
        non_seqs = [questions, m_1]
        params_keys = ['U', 'W1', 'b1', 'W2', 'b2', 'Wb', 'Wf', 'Uf', 'bf',
                       'Wm',
                       'bm', 'dim']
        non_seqs = non_seqs + [params[k] for k in params_keys]

        memories, updates = theano.scan(compute_memory,
                                        sequences=seqs,
                                        outputs_info=init_states,
                                        non_sequences=non_seqs,
                                        name="gru_layer",
                                        n_steps=nb_passes,
                                        strict=True)
        return memories
