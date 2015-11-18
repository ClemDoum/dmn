import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import InputLayer

from dmn.episodic_memory import EpisodicMemoryLayer


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
    hidden_n_1 = layer.inner_gru_step(input_n, hid_previous, attention_gate)
    inputs = [input_n, mask_n, hid_previous, attention_gate, W_hid_stacked,
              W_in_stacked, b_stacked]
    fn = theano.function(inputs, [hidden_n_1], mode='DebugMode')
    return fn


if __name__ == '__main__':
    floatX = "float32"
    intX = "int32"
    # Data
    batch_size = 10
    fact_length = 3
    question_length = 7

    representation_dim = 40

    n_decodesteps = 100
    # nb_passes = 3

    fact = T.matrix("fact", dtype=floatX)
    facts = T.tensor3("facts", dtype=floatX)
    question = T.matrix(name="question", dtype=floatX)
    memory = T.matrix("memory", dtype=floatX)
    gate = T.matrix("gate", dtype=floatX)
    mask = T.vector("mask", dtype=intX)
    W_hid = T.matrix("W_hid", dtype=floatX)
    W_in = T.matrix("W_in", dtype=floatX)
    b = T.vector("b", dtype=floatX)

    l_in = InputLayer(shape=(batch_size, fact_length, representation_dim),
                      input_var=facts)
    layer = EpisodicMemoryLayer(l_in, representation_dim, n_decodesteps)

    input_n = np.random.normal(
        size=(batch_size, 3 * representation_dim)).astype(
        floatX)
    mask_n = np.random.binomial(n=1, p=.8,
                                size=(batch_size, fact_length)).astype(intX)
    hid_previous = np.random.normal(
        size=(batch_size, representation_dim)).astype(floatX)
    attention_gate = np.random.normal(
        size=(batch_size, representation_dim)).astype(floatX)
    W_hid_stacked = np.random.normal(
        size=(representation_dim, 3 * representation_dim)).astype(floatX)
    W_in_stacked = np.random.normal(
        size=(representation_dim, 3 * representation_dim)).astype(floatX)
    b_stacked = np.random.normal(size=(3 * representation_dim,)).astype(floatX)

    # # Compute the gates
    # fn = gating_function(layer, first_fact, question, memory, representation_dim)
    # gates = fn(ff, m, q)
    # print gates
    # print gates[0].shape

    # Test the modified GRU
    # Compute the gates
    get_episode = test_inner_gru_step(layer, fact, memory, gate, W_hid, W_in,
                                      b)
    episode = get_episode(input_n, hid_previous, attention_gate, W_hid_stacked,
                          W_in_stacked, b_stacked)
    print episode
    print episode.shape
    # test_masked_inner_gru_step(layer, first_fact, mask, question, memory)
