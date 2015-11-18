import lasagne
import lasagne.init as init
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import MergeLayer, GRULayer


class episodic_memory_layer(MergeLayer):
    def __init__(self, incoming, num_units, W_hid_to_cell=init.Constant(0.1),
                 b_cell=init.Constant(0.), mask_input=None, **kwargs):
        incomings = [incoming]
        if mask_input:
            incomings.append(mask_input)
        super(episodic_memory_layer, self).__init__(incomings, **kwargs)
        mask_input = mask_input[1] if mask_input else None
        self.num_units = num_units
        self.gru_layer = GRULayer(incomings[0], num_units=num_units,
                                  mask_input=mask_input, **kwargs)
        for param in self.gru_layer.get_params():
            self.add_param(param, param.shape, param.name)
        # self.add_param(self.gru_layer.get_params())
        self.W_hid_to_cell = self.add_param(
            W_hid_to_cell, (num_units, num_units), name="W_hid_to_cell")
        self.b_cell = self.add_param(
            b_cell, (num_units,), name="b_cell", regularizable=False)


        self.params = self.gru_layer.params

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, **kwargs):
        gru_output = self.gru_layer.get_output_for(inputs=inputs,
                                                   **kwargs)
        output = T.dot(gru_output[:, -1, :], self.W_hid_to_cell) + self.b_cell
        return T.tanh(output)


def main():
    floatX = "float32"
    intX = "int32"
    input_var = T.tensor3('inputs', dtype=floatX)
    target_var = T.ivector('targets')

    question_length = 60
    question_dim = 13
    output_dim = 1
    batch_size = 8

    network = lasagne.layers.InputLayer(
        shape=(None, question_length, question_dim),
        input_var=input_var)

    network = episodic_memory_layer(network, question_dim)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    print params
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    inputs = np.random.normal(
        size=(batch_size, question_length, question_dim)).astype(floatX)
    targets = np.random.binomial(
        1, p=.3, size=(batch_size,)).astype(intX)

    for epoch in range(1000):
        loss = train_fn(inputs, targets)
        print "loss: %f" % loss


if __name__ == '__main__':
    main()
