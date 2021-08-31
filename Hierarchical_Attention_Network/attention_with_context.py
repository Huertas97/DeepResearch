from keras.preprocessing.text import Tokenizer,  text_to_word_sequence
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed
from keras import backend as K
from keras import optimizers
from keras.models import Model
def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`. 
        samples = nº docs or nº of sentences (it depends on the level we are)
        steps = nº of sentences per doc or nº of words per sentence
        features = dimensions of the sentence or word embeddings
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)
    
    
    # In the Keras API, we recommend creating layer weights in the build(self, inputs_shape) method of your layer
    # https://keras.io/guides/making_new_layers_and_models_via_subclassing/
    # https://www.tutorialspoint.com/keras/keras_customized_layer.htm
    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    
    # The __call__() method of your layer will automatically run build the first time it is called. 
    # https://keras.io/guides/making_new_layers_and_models_via_subclassing/
    # https://www.tutorialspoint.com/keras/keras_customized_layer.htm
    def call(self, x, mask=None):
        # x --> 3D tensor with shape: `(samples, steps, features)`. 
        # Following the formulas of HAN paper (Hierarchical Attention Networks for Document Classification)
        
        # FORMULA 5
        # create the hidden representation of the hidden state (we will call it 'original')
        uit = dot_product(x, self.W) 
        
        # if bias is included added to the hidden representation of the hidden state
        # Just how a single dense layer works
        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        
        # FORMULA 6
        ait = dot_product(uit, self.u)

        a = K.exp(ait) # numerator of the softmax function

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx()) # we select the masking positions and apply them to the numerator

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        # cast change th dtype. Here we are completing the softmax function. Divide the numerator by the denominator
        # the denomimnator is the summatory of the numerator (look at the softmax formula)
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx()) 
        
        a = K.expand_dims(a) # Añade una dimension
        
        # FORMULA 7
        # Now we multiply each 'original' hidden state by its attention value
        # Then we add all this weighted values. We are weighting the original 
        # hidden states by it importance. This is what attention consists of. 
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
