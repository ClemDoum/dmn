from lasagne.layers.recurrent import GRULayer
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import InputLayer

from dmn.episodic_memory import EpisodicMemoryLayer, concatenate


# theano.config.exception_verbosity = 'high'

def gating_function(layer, fact, memory, question, representation_dim):
    incoming = [fact, question, memory]
    gating_layer = layer.attention_gate(incoming, representation_dim)
    gates = gating_layer.get_output_for(incoming)
    fn = theano.function(incoming, [gates], mode='DebugMode')
    return fn


def test_inner_gru_step(layer, input_n, hid_previous, attention_gate,
                        W_hid_stacked, W_in_stacked, b_stacked):
    hidden_n_1 = layer.inner_gru_step(input_n, hid_previous, attention_gate,
                                      W_hid_stacked, W_in_stacked, b_stacked)
    inputs = [input_n, hid_previous, attention_gate, W_hid_stacked,
              W_in_stacked, b_stacked]
    fn = theano.function(inputs, hidden_n_1, mode='DebugMode',
                         on_unused_input='ignore')
    return fn


def test_masked_inner_gru_step(layer, input_n, mask_n, hid_previous,
                               attention_gate, W_hid_stacked, W_in_stacked,
                               b_stacked):
    hidden_n_1 = layer.masked_inner_gru_step(input_n, mask_n, hid_previous,
                                             attention_gate, W_hid_stacked,
                                             W_in_stacked, b_stacked)
    inputs = [input_n, mask_n, hid_previous, attention_gate, W_hid_stacked,
              W_in_stacked, b_stacked]
    fn = theano.function(inputs, hidden_n_1, mode='DebugMode',
                         on_unused_input='ignore')
    return fn


def test_gru_episode(layer, inputs, gates, initial_episode, W_hid_stacked,
                     W_in_stacked, b_stacked):
    ep = layer.episode(inputs, gates, initial_episode, W_hid_stacked,
                       W_in_stacked, b_stacked, mask=None)

    ins = [inputs, gates, initial_episode, W_hid_stacked, W_in_stacked,
           b_stacked]
    fn = theano.function(ins, ep, mode='DebugMode', on_unused_input='ignore')
    return fn


def test_masked_gru_episode():
    pass


if __name__ == '__main__':
    floatX = "float32"
    intX = "int32"
    # Data
    batch_size = 10
    fact_length = 3
    question_length = 7

    representation_dim = 38

    n_decodesteps = 100
    # nb_passes = 3

    fact = T.matrix("fact", dtype=floatX)
    facts = T.tensor3("facts", dtype=floatX)
    masks = T.matrix("masks", dtype=intX)
    questions = T.matrix(name="question", dtype=floatX)
    memory = T.matrix("memory", dtype=floatX)
    gate = T.matrix("gate", dtype=floatX)
    mask = T.vector("mask", dtype=intX)
    shuffled_broadcastable_mask = mask.dimshuffle(0, 'x')
    W_hid = T.matrix("W_hid", dtype=floatX)
    W_in = T.matrix("W_in", dtype=floatX)
    b = T.vector("b", dtype=floatX)

    l_in = InputLayer(shape=(batch_size, fact_length, representation_dim),
                      input_var=facts)
    l_mask = InputLayer(shape=(batch_size, fact_length),
                        input_var=masks)
    layer = EpisodicMemoryLayer(l_in, representation_dim, n_decodesteps)
    layer_masked = EpisodicMemoryLayer(l_in, representation_dim, n_decodesteps,
                                       mask_input=l_mask)

    np_facts = np.random.normal(
        size=(batch_size, fact_length, representation_dim)).astype(floatX)
    inputs = np.random.normal(
        size=(batch_size, fact_length, 3 * representation_dim)).astype(
        floatX)
    input_n = np.random.normal(
        size=(batch_size, 3 * representation_dim)).astype(
        floatX)
    mask_n = np.random.binomial(n=1, p=.8,
                                size=(batch_size,)).astype(intX)
    shuffled_broadcastable_mask_n = mask_n.reshape((mask_n.shape[0], 1))
    hid_previous = np.random.normal(
        size=(batch_size, representation_dim)).astype(floatX)

    W_hid_stacked = np.random.normal(
        size=(representation_dim, 3 * representation_dim)).astype(floatX)
    W_in_stacked = np.random.normal(
        size=(representation_dim, 3 * representation_dim)).astype(floatX)
    b_stacked = np.random.normal(size=(3 * representation_dim,)).astype(floatX)
    np_questions = np.random.normal(
        size=(batch_size, representation_dim)).astype(floatX)
    np_memory = np.random.normal(
        size=(batch_size, representation_dim)).astype(floatX)


    # # Compute the gates
    # fn = gating_function(layer, first_fact, question, memory, representation_dim)
    # gates = fn(ff, m, q)
    # print gates
    # print gates[0].shape

    # # Test the modified GRU
    # get_inner_gru_state = test_inner_gru_step(
    #     layer, fact, memory, gate, W_hid, W_in, b)
    # inner_gru_states = get_inner_gru_state(
    #     input_n, hid_previous, attention_gate, W_hid_stacked, W_in_stacked,
    #     b_stacked)
    # print inner_gru_states
    # print inner_gru_states.shape
    #
    # # Test the modified GRU with a mask
    #
    # get_inner_gru_state_with_mask = test_masked_inner_gru_step(
    #
    #     layer, fact, shuffled_broadcastable_mask, memory, gate, W_hid, W_in, b)
    #
    # masked_inner_gru_state = get_inner_gru_state_with_mask(
    #     input_n, shuffled_broadcastable_mask_n, hid_previous, attention_gate,
    #     W_hid_stacked, W_in_stacked, b_stacked)
    # print masked_inner_gru_state
    # print masked_inner_gru_state.shape

    # # Test
    episodic_mem = layer.get_output_for([facts, questions])
    fn = theano.function([facts, questions], episodic_mem,  # mode='DebugMode',
                         on_unused_input='ignore')

    np_episodic_mem = fn(np_facts, np_questions)
    print np_episodic_mem
    print np_episodic_mem.shape
