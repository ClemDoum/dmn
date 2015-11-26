import theano
import theano.tensor as T
from lasagne import init, nonlinearities
from lasagne.layers import MergeLayer, Gate
from theano.gradient import np


# concatenate funciton from Kyunghyun Cho
# https://github.com/kyunghyuncho/dl4mt-material

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


class EpisodicMemoryLayer(MergeLayer):
    def __init__(self,
                 incoming,
                 num_units,
                 n_decodesteps,
                 attention_mechanism='gru',
                 inner_resetgate=Gate(W_cell=None),
                 inner_updategate=Gate(W_cell=None),
                 inner_hidden_update=Gate(W_cell=None,
                                          nonlinearity=nonlinearities.tanh),
                 outer_resetgate=Gate(W_cell=None),
                 outer_updategate=Gate(W_cell=None),
                 outer_hidden_update=Gate(W_cell=None,
                                          nonlinearity=nonlinearities.tanh),
                 hid_init=init.Constant(0.),
                 grad_clipping=False,
                 gradient_steps=-1,
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
        self.n_decodesteps = n_decodesteps
        self.grad_clipping = grad_clipping
        self.precompute_input = precompute_input
        self.gradient_steps = gradient_steps

        mechanisms = ['gru', 'softmax']
        if attention_mechanism not in mechanisms:
            raise ValueError("Invalid attention_mechanism.\n"
                             "antention_mechanism must be in %s.\n"
                             "Found %s." % (mechanisms, attention_mechanism))
        self.attention_mechanism = attention_mechanism

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name, location):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (
                self.add_param(gate.W_in, (num_inputs, num_units),
                               name="{}_W_in_to_{}".format(gate_name,
                                                           location)),
                self.add_param(gate.W_hid, (num_units, num_units),
                               name="{}_W_hid_to_{}".format(gate_name,
                                                            location)),
                self.add_param(gate.b, (num_units,),
                               name="{}_b_{}".format(gate_name, location),
                               regularizable=False),
                gate.nonlinearity)

        # Add in all parameters from inner gates
        (self.inner_W_in_to_updategate, self.inner_Wi_hid_to_updategate,
         self.inner_b_updategate,
         self.inner_nonlinearity_updategate) = add_gate_params(
            inner_updategate, 'updategate', 'inner')
        (self.inner_W_in_to_resetgate,
         self.inner_W_hid_to_resetgate,
         self.inner_b_resetgate,
         self.inner_nonlinearity_resetgate) = add_gate_params(inner_resetgate,
                                                              'resetgate',
                                                              'inner')

        (self.inner_W_in_to_hidden_update,
         self.inner_W_hid_to_hidden_update,
         self.inner_b_hidden_update,
         self.inner_nonlinearity_hid) = add_gate_params(
            inner_hidden_update, 'hidden_update', 'inner')

        # Add in all parameters from outer gates
        (self.outer_W_in_to_updategate,
         self.outer_Wi_hid_to_updategate,
         self.outer_b_updategate,
         self.outer_nonlinearity_updategate) = add_gate_params(
            outer_updategate,
            'updategate', 'outer')
        (self.outer_W_in_to_resetgate,
         self.outer_W_hid_to_resetgate,
         self.outer_b_resetgate,
         self.outer_nonlinearity_resetgate) = add_gate_params(outer_resetgate,
                                                              'resetgate',
                                                              'outer')

        (self.outer_W_in_to_hidden_update,
         self.outer_W_hid_to_hidden_update,
         self.outer_b_hidden_update,
         self.outer_nonlinearity_hid) = add_gate_params(
            outer_hidden_update, 'hidden_update', 'outer')

    # At each call to scan, input_n will be (n_time_steps, 3*num_units).
    # We define a slicing function that extract the input to each GRU gate
    def slice_w(self, x, n):
        return x[:, n * self.num_units:(n + 1) * self.num_units]

    def inner_gru_step(self, input_n, attention_gate, hid_previous,
                       W_hid_stacked, W_in_stacked, b_stacked):
        hid = self.gru_step(input_n, hid_previous, W_hid_stacked, W_in_stacked,
                            b_stacked)
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

    def gru_step(self, input_n, hid_previous, W_hid_stacked, W_in_stacked,
                 b_stacked, outer=False):
        if outer:
            nonlinearity_resetgate = self.outer_nonlinearity_resetgate
            nonlinearity_updategate = self.outer_nonlinearity_updategate
            nonlinearity_hid = self.outer_nonlinearity_hid
        else:
            nonlinearity_resetgate = self.inner_nonlinearity_resetgate
            nonlinearity_updategate = self.inner_nonlinearity_updategate
            nonlinearity_hid = self.inner_nonlinearity_hid

        # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
        hid_input = T.dot(hid_previous, W_hid_stacked)

        if self.grad_clipping is not False:
            input_n = theano.gradient.grad_clip(
                input_n, -self.grad_clipping, self.grad_clipping)
            hid_input = theano.gradient.grad_clip(
                hid_input, -self.grad_clipping, self.grad_clipping)

        if not self.precompute_input or outer:
            # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
            input_n = T.dot(input_n, W_in_stacked) + b_stacked

        # Reset and update gates
        resetgate = self.slice_w(hid_input, 0) + self.slice_w(input_n, 0)
        updategate = self.slice_w(hid_input, 1) + self.slice_w(input_n, 1)
        resetgate = nonlinearity_resetgate(resetgate)
        updategate = nonlinearity_updategate(updategate)

        # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
        hidden_update_in = self.slice_w(input_n, 2)
        hidden_update_hid = self.slice_w(hid_input, 2)
        hidden_update = hidden_update_in + resetgate * hidden_update_hid
        if self.grad_clipping is not False:
            hidden_update = theano.gradient.grad_clip(
                hidden_update, -self.grad_clipping, self.grad_clipping)
        hidden_update = nonlinearity_hid(hidden_update)
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

    def episode(self, inputs, gates, initial_episode, W_hid_stacked,
                W_in_stacked, b_stacked, mask=None):
        # Add a broadcastable dimension to gates (n_steps, batch_size)
        # in order to perform the elementwise product with the hidden
        # states of shape (n_steps, batch_size, hidden_dim)

        # gates = gates.dimshuffle(0, 'x')


        if self.attention_mechanism == 'gru':
            if mask is not None:
                sequences = [inputs, mask, gates]
                step_fun = self.masked_inner_gru_step
            else:
                sequences = [inputs, gates]
                step_fun = self.inner_gru_step

            # The hidden-to-hidden weight matrix is always used in step
            non_seqs = [W_hid_stacked]
            # When we aren't precomputing the input outside of scan, we need to
            # provide the input weights and biases to the step function
            if not self.precompute_input:
                non_seqs += [W_in_stacked, b_stacked]
            # theano.scan only allows for positional arguments, so when
            # self.precompute_input is True, we need to supply fake placeholder
            # arguments for the input weights and biases.
            else:
                non_seqs += [(), ()]

            hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                # TODO: check if the GRU is reset to 0 at each pass
                outputs_info=[initial_episode],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]
            return hid_out[-1]

        elif self.attention_mechanism == 'softmax':
            # TODO: implement softmax attention + gates supervision
            return NotImplementedError
        else:
            raise ValueError(
                "Input sequence length cannot be specified as "
                "None when unroll_scan is True")

    def episodic_memory(self, episode_n, mem_previous, W_hid_stacked,
                        W_in_stacked, b_stacked, mask=None):
        mem = self.gru_step(episode_n, mem_previous, W_hid_stacked,
                            W_in_stacked, b_stacked, outer=True)
        return mem

    def attention_gate(self, facts, memory, question):
        # TODO: for the first iteration question and memory are the same so
        # we can speedup the computation

        # facts is (num_batch * fact_length * memory_dim)
        # questions is (num_batch * memory_dim)
        # memory is (num_batch * memory_dim)
        # attention_gates must be (fact_length * nb_batch * 1)

        # Compute z (num_batch * fact_length * (7*memory_dim + 2))

        # Dimshuffle facts to get a shape of
        # (fact_length * num_batch * memory_dim)
        facts = facts.dimshuffle(1, 0, 2)

        # Pad questions and memory to be of shape
        # (_ * num_batch * memory_dim)
        memory = T.shape_padleft(memory)
        question = T.shape_padleft(question)

        to_concatenate = list()
        to_concatenate.extend([facts, memory, question])
        to_concatenate.extend([facts * question, facts * memory])
        to_concatenate.extend([T.abs_(facts - question),
                               T.abs_(facts - memory)])

        # z = concatenate(to_concatenate, axis=2)

        # TODO: to be continued for the moment just return ones
        return T.ones((facts.shape[1], facts.shape[0], 1))

    def get_output_shape_for(self, input_shapes):
        NotImplementedError

    def get_output_for(self, inputs, **kwargs):
        # Retrieve the facts
        facts = inputs[0]

        # return facts

        # Retrieve the mask when it is supplied and the questions
        if len(inputs) > 2:
            mask = inputs[1]
            questions = inputs[2]
        else:
            mask = None
            questions = inputs[1]

        # Treat all dimensions after the second as flattened feature dimensions
        if facts.ndim > 3:
            facts = T.flatten(facts, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        facts = facts.dimshuffle(1, 0, 2)
        if mask:
            mask.dimshuffle(1, 0)
        seq_len, num_batch, _ = facts.shape

        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        inner_W_in_stacked = T.concatenate(
            [self.inner_W_in_to_resetgate, self.inner_W_in_to_updategate,
             self.inner_W_in_to_hidden_update], axis=1)

        # Same for hidden weight matrices
        inner_W_hid_stacked = T.concatenate(
            [self.inner_W_hid_to_resetgate, self.inner_Wi_hid_to_updategate,
             self.inner_W_hid_to_hidden_update], axis=1)

        # Stack gate biases into a (3*num_units) vector
        inner_b_stacked = T.concatenate(
            [self.inner_b_resetgate, self.inner_b_updategate,
             self.inner_b_hidden_update], axis=0)

        outer_W_in_stacked = T.concatenate(
            [self.outer_W_in_to_resetgate, self.outer_W_in_to_updategate,
             self.outer_W_in_to_hidden_update], axis=1)

        # Same for hidden weight matrices
        outer_W_hid_stacked = T.concatenate(
            [self.outer_W_hid_to_resetgate, self.outer_Wi_hid_to_updategate,
             self.outer_W_hid_to_hidden_update], axis=1)

        # Stack gate biases into a (3*num_units) vector
        outer_b_stacked = T.concatenate(
            [self.outer_b_resetgate, self.outer_b_updategate,
             self.outer_b_hidden_update], axis=0)

        # Define facts for gates as we may change the facts shape
        # to precompute the input of the GRU
        facts_for_gates = facts

        if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            facts = T.dot(facts, inner_W_in_stacked) + inner_b_stacked
            # We'll have to use both precomputed facts and facts_for_gates
            non_seqs = [facts, facts_for_gates, questions]
        else:
            # We do not need to work with the facts_for_gates variable
            non_seqs = [facts, facts, questions]

        # Set the inial memory to the question
        episodic_memory = questions.copy()  # is the copy necessary?

        non_seqs += [inner_W_hid_stacked, inner_W_in_stacked, inner_b_stacked,
                     outer_W_hid_stacked, outer_W_in_stacked, outer_b_stacked]

        def step(q, f, f_for_gates, e_m, i_W_hid_stacked, i_W_in_stacked,
                 i_b_stacked, o_W_hid_stacked, o_W_in_stacked, o_b_stacked):
            # 1. Compute the gates with the facts, the current memory and
            #  the question

            # We create a copy of e_m because it's modified by the attention
            # computation
            e_m_for_gates = e_m.copy()
            gates = self.attention_gate(f_for_gates, e_m_for_gates, q)

            # 2. Compute the episodes from the facts, the gates and
            # the previous memory

            # We create a copy of f because it's modified by episode
            # computation
            f_for_episodes = f.copy()
            episodes = self.episode(f_for_episodes, gates, q, i_W_hid_stacked,
                                    i_W_in_stacked, i_b_stacked)

            # 3. Compute the new episodic memory from the new episode and
            # the previous memory
            mem = self.episodic_memory(episodes, e_m, o_W_hid_stacked,
                                       o_W_in_stacked, o_b_stacked)
            return mem

        # Get a new episodic memory n_decodesteps times
        outs, _ = theano.scan(
            fn=step,
            outputs_info=[episodic_memory],
            non_sequences=non_seqs,
            truncate_gradient=self.gradient_steps,
            strict=True,
            n_steps=self.n_decodesteps
        )
        # Keep only the last memory
        return outs[-1]
