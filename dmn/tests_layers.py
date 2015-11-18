import numpy as np
import theano
import theano.tensor as T

from episodic_memory import GatingLayer


def gating_function(fact, memory, question, representation_dim):
    incoming = [fact, question, memory]
    gating_layer = layer.attention_gate(incoming, representation_dim)
    gates = gating_layer.get_output_for(incoming)
    fn = theano.function(incoming, [gates], mode='DebugMode',
                         on_unused_input='ignore')
    return fn



if __name__ == '__main__':
    floatX = "float32"
    # Data
    batch_size = 10
    fact_length = 15
    question_length = 20

    representation_dim = 40

    # nb_passes = 3

    fact = T.tensor3(dtype=floatX)
    first_fact = T.matrix(dtype=floatX)
    question = T.matrix(dtype=floatX)
    memory = T.matrix(dtype=floatX)

    f = np.random.normal(
        size=(batch_size, fact_length, representation_dim)).astype(floatX)
    ff = np.random.normal(size=(batch_size, representation_dim)).astype(floatX)
    q = np.random.normal(size=(batch_size, representation_dim)).astype(floatX)
    m = q.copy()

    # Compute the gates
    fn = gating_function(first_fact, question, memory, representation_dim)
    gates = fn(ff, m, q)
    print gates
    print gates[0].shape

