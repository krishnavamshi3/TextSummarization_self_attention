import tensorflow as tf
from utils import _calc_final_dist
from layers import LstmEncoder, GruEncoder, LstmDecoder, GruDecoder, BahdanauAttention, Pointer, MultiHeadAttention, PSelfVocab

class PGN(tf.keras.Model):
  
  def __init__(self, encoder, decoder, params):
    super(PGN, self).__init__()
    self.params = params
    if encoder == "LSTM":
        self.encoder = LstmEncoder(params["vocab_size"], params["embed_size"], params["enc_units"], params["batch_size"])
    else:
        self.encoder = GruEncoder(params["vocab_size"], params["embed_size"], params["enc_units"], params["batch_size"])
    self.attention = BahdanauAttention(params["attn_units"])
    if decoder == "LSTM":
        self.decoder = LstmDecoder(params["vocab_size"], params["embed_size"], params["dec_units"], params["batch_size"])
    else:
        self.decoder = GruDecoder(params["vocab_size"], params["embed_size"], params["dec_units"], params["batch_size"])
    self.pointer = Pointer()
    #self.pVocab = Pvocab(params["vocab_size"])
    self.self_attention = MultiHeadAttention(d_model=params["enc_units"], num_heads= 4,)
    self.pSelfVocab = PSelfVocab(params["vocab_size"])
    
  def call_encoder(self, enc_inp):
    enc_hidden = self.encoder.initialize_hidden_state()
    enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)
    return enc_hidden, enc_output
    
  def call(self, enc_output, dec_hidden, enc_inp, enc_extended_inp,  dec_inp, batch_oov_len):
    
    predictions = []
    self_predictions = []
    attentions = []
    p_gens = []
    context_vector, _ = self.attention(dec_hidden, enc_output)
    y = tf.random.uniform((self.params["batch_size"], self.params["max_enc_len"], self.params["enc_units"]))  # (batch_size, encoder_sequence, d_model)
    mha_out, mha_attn_weights = self.self_attention(y, k=y, q=y, mask=None) # (batch_size, encoder_sequence, d_model)
    self_context_vector = tf.reduce_sum(mha_out, 1) # SUM to compress word information to single tensor.
    for t in range(dec_inp.shape[1]):
      # print("Context vector : ")
      # print(context_vector)
      # print("Self context vector : ")
      # print(self_context_vector)
      output = tf.concat([context_vector, self_context_vector], axis=-1)
      # print("concatenate vector : ")
      # print(output)
      dec_x, pred,dec_hidden = self.decoder(tf.expand_dims(dec_inp[:, t],1), dec_hidden, enc_output, output)
      context_vector, attn = self.attention(dec_hidden, enc_output)
      # self_pred = self.pVocab(self_context_vector, dec_hidden)
      p_gen = self.pointer(context_vector, dec_hidden, tf.squeeze(dec_x, axis=1))
      # output = tf.concat([tf.expand_dims(context_vector, 1), tf.expand_dims(self_context_vector, 1)], axis=-1)
      # output = tf.concat([tf.expand_dims(output, 1), tf.expand_dims(dec_hidden, 1)], axis=-1)
      
      # output = tf.reshape(output, (-1, output.shape[2]))
      # pred = self.pSelfVocab(output) # combining the context vector from self attention and softattention
      predictions.append(pred)
      # self_predictions.append(self_pred)
      attentions.append(attn)
      p_gens.append(p_gen)
    final_dists = _calc_final_dist( enc_extended_inp, predictions, attentions, p_gens, batch_oov_len, self.params["vocab_size"], self.params["batch_size"])
    if self.params["mode"] == "train":
      return tf.stack(final_dists, 1), dec_hidden  # predictions_shape = (batch_size, dec_len, vocab_size) with dec_len = 1 in pred mode
    else:
      return tf.stack(final_dists, 1), dec_hidden, context_vector, tf.stack(attentions, 1), tf.stack(p_gens, 1)
  