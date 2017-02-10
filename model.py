from seq2seq.models import Seq2Seq
from recurrentshop.engine import RecurrentContainer
from seq2seq.cells import LSTMCell
from seq2seq.cells import LSTMDecoderCell
from keras.layers import Input, Embedding, Dropout, TimeDistributed, Dense
from keras import backend as K
from keras.models import Model


def build_decoder(dropout, embed_dim, output_length, output_dim, peek=False, unroll=False):
    decoder = RecurrentContainer(
        readout='add' if peek else 'readout_only', output_length=output_length, unroll=unroll, decode=True,
        input_length=shape[1]
    )
    # for i in range(depth[1]):
    decoder.add(Dropout(dropout, batch_input_shape=(None, embed_dim)))
    decoder.add(LSTMDecoderCell(output_dim=output_dim, hidden_dim=embed_dim, batch_input_shape=(shape[0], embed_dim)))

    return decoder


def SkipThoughtModel(
        sent_len, vocab_size, embed_dims, output_length, output_dim, dropout=0.4, unroll=False, teacher_force=False
):
    input_sent = Input(shape=(sent_len, vocab_size), dtype=K.floatx())
    input_sent._keras_history[0].supports_masking = True

    encoder = RecurrentContainer(
        readout=True, input_length=sent_len, unroll=unroll, stateful=False
    )
    # for i in range(depth[0]):
    encoder.add(LSTMCell(embed_dims, batch_input_shape=(None, embed_dims)))
    encoder.add(Dropout(dropout))

    dense1 = TimeDistributed(Dense(embed_dims))
    dense1.supports_masking = True
    dense2 = Dense(embed_dims)

    encoded_seq = dense1(input)
    encoded_seq = encoder(encoded_seq)

    states = [None] * 2
    encoded_seq = dense2(encoded_seq)
    inputs = [input]
    if teacher_force:
        truth_tensor_prev = Input(batch_shape=(None, output_length, output_dim))
        truth_tensor_prev._keras_history[0].supports_masking = True
        truth_tensor_next = Input(batch_shape=(None, output_length, output_dim))
        truth_tensor_next._keras_history[0].supports_masking = True
        inputs += [truth_tensor_prev, truth_tensor_next]

    prev_decoder = build_decoder(dropout=dropout, unroll=unroll, output_length=output_length)
    next_decoder = build_decoder()
    prev_decoded_seq = prev_decoder(
        {'input': encoded_seq, 'ground_truth': inputs[1] if teacher_force else None, 'initial_readout': encoded_seq,
         'states': states})

    next_decoded_seq = next_decoder(
        {'input': encoded_seq, 'ground_truth': inputs[2] if teacher_force else None, 'initial_readout': encoded_seq,
         'states': states})

    model = Model(inputs, [prev_decoded_seq, next_decoded_seq])
    model.encoder = encoder
    model.decoders = [prev_decoder, next_decoder]
    return model