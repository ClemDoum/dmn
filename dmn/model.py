from theano import tensor as T

import theano
from theano.gradient import np
from theano.tensor.nnet import sigmoid

floatX = "float32"


def slice(tensor, idx, dim):
    if dim == 2:
        return tensor[:, idx * dim:(idx + 1) * dim]
    elif dim == 3:
        return tensor[:, :, idx * dim:(idx + 1) * dim]
    else:
        raise NotImplementedError


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


def compute_attention(facts, questions, memories, W_1, b_1, W_2, b_2, W_b,
                      mask=None):
    # Add the timestep dimension to the questions and the memories
    m = memories
    q = questions
    # Compute z
    to_concatenate = list()
    to_concatenate.extend([facts, m, q])
    to_concatenate.extend([facts * q, facts * m])
    to_concatenate.extend([T.abs_(facts - q), T.abs_(facts - m)])
    to_concatenate.extend([T.dot(facts.T, T.dot(W_b, q)),
                           T.dot(facts.T, T.dot(W_b, m))])
    z = concatenate(to_concatenate, axis=axis)
    # Compute the gates
    gates = T.dot(W_2, T.tanh(T.dot(W_1, z) + b_1)) + b_2
    gates = T.nnet.sigmoid(gates)
    return gates


def compute_modified_gru_state(g, x_dot_W_plus_b, mask, h_1, U, dim):
    h = compute_gru_state(x_dot_W_plus_b, mask, h_1, U, dim)
    h = g * h + (1 - g) * h_1
    h = apply_mask_to_gru_state(h, h_1, mask)
    return h


def compute_episodes(attentions, facts, questions, Wf, Uf, bf, dim, mask=None):
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


def get_episodic_memory(facts, questions, nb_passes):

    def compute_memory(f, q, m_1, U, W_1, b_1, W_2, b_2, W_b, Wf, Uf, bf, Wm,
                       Bm, dim):
        # 1. Compute the attention from the fact, memory and question
        a = compute_attention(f, q, m_1, W_1, b_1, W_2, b_2, W_b)
        # 2. Compute the episode from the attention and the facts
        e = compute_episodes(a, f, q, Wf, Uf, bf, dim)
        # 3. Compute the new memory from the episode
        e_dot_Wm_plus_bm = T.dot(e, Wm) + Bm
        m = compute_gru_state(e_dot_Wm_plus_bm, m_1, U, dim)
        return m

    seqs = [facts]
    init_states = [T.alloc(questions)]  # Initialize it to the question
    non_seqs = [questions]

    memories, updates = theano.scan(compute_memory,
                                    sequences=seqs,
                                    outputs_info=init_states,
                                    non_sequences=non_seqs,
                                    name="gru_layer",
                                    n_steps=nb_passes,
                                    strict=True)
    return memories


def compute_gru_state(x_dot_W_plus_b, mask, h_1, U, dim):
    h_1_dot_U = T.dot(h_1, U)

    r = sigmoid(slice(x_dot_W_plus_b, 0, dim) + slice(h_1_dot_U, 0, dim))
    u = sigmoid(slice(x_dot_W_plus_b, 1, dim) + slice(h_1_dot_U, 1, dim))

    h = T.tanh(slice(x_dot_W_plus_b, 2, dim) + r * slice(h_1_dot_U, 2, dim))

    h = u * h_1 + (1 - u) * h

    return h


def apply_mask_to_gru_state(h, h_1, mask):
    if mask:
        h = mask[:, None] * h + (1 - mask[:, None]) * h_1
    return h


def compute_gru_state_with_mask(x_dot_W_plus_b, mask, h_1, U, dim):
    h = compute_gru_state(x_dot_W_plus_b, h_1, U, dim)
    h = apply_mask_to_gru_state(h, h_1, mask)
    return h


def gru_layer(x, W, U, b, init_states=None, mask=None):
    dim = U.shape[1]
    if x.ndim == 3:
        n_samples = x.shape[1]
    else:
        n_samples = 1

    x_dot_W_plus_b = T.dot(x, W) + b
    if mask is None:
        mask = T.alloc(1., x.shape[0], 1)

    seqs = [x_dot_W_plus_b, mask]
    if init_states is None:
        init_states = [T.alloc(0., n_samples, dim)]
    non_seqs = [U, dim]

    states, updates = theano.scan(compute_gru_state_with_mask,
                                  sequences=seqs,
                                  outputs_info=init_states,
                                  non_sequences=non_seqs,
                                  name="gru_layer",
                                  strict=True)
    return [states]


def compute_episodic_memory():
    batch_size = 10
    fact_length = 15
    question_length = 20

    representation_dim = 40

    nb_passes = 3

    facts = T.tensor3(dtype=floatX)
    questions = T.tensor3(dtype=floatX)

    episodic_memories = get_episodic_memory(facts, questions, nb_passes)

    fn = theano.function([facts, questions], [episodic_memories])

    f = np.random.normal(size=(batch_size, fact_length, representation_dim))
    q = np.random.normal(
        size=(batch_size, question_length, representation_dim))

    m = fn([f, q])

    print m.shape
    print m


if __name__ == '__main__':
    compute_episodic_memory()
