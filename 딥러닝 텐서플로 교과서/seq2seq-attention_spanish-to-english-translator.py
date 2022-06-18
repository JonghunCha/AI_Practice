"""
os : windows 10
gpu : RTX 3060
tensorflow : 2.4.0
python : 3.8
cuda : 11.0
cudnn : 8.0.5
anaconda : 4.11.0

-데이터-
영어와 그에 대응하는 스페인어 묶음이 약 12만여개로 이루어져 있음

-목적-
스페인어를 입력으로 받아 영어로 번역하는 seq2seq모델 학습

-학습 결과-
초기 상태의 Loss는 1.4648이었으나, 학습 후 약 0.03~0.04정도로 수렴되었음

또한 attention을 시각화하여 attention이 중요한 단어에 집중하는 것을 확인할 수 있었음
"""
import tensorflow
import unicodedata
import re
import io
import os
import time
import numpy
import matplotlib.pyplot
import matplotlib.ticker
from sklearn.model_selection import train_test_split

def unicode_to_ascii(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    w = "<start> " + w + " <end>"
    return w

def create_dataset(path, num_examples):
    lines = io.open(path, encoding="UTF-8").read().strip().split("\n")
    word_pairs = [[preprocess_sentence(w) for w in l.split("\t")] for l in lines[:num_examples]]
    return zip(*word_pairs)

def max_length(tensor):
    return max(len(t) for t in tensor)

def tokenize(lang):
    lang_tokenizer = tensorflow.keras.preprocessing.text.Tokenizer(filters="")
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tensorflow.keras.preprocessing.sequence.pad_sequences(tensor, padding="post")
    return tensor, lang_tokenizer

def load_dataset(path, num_examples=None):
    targ_lang, inp_lang = create_dataset(path, num_examples)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

class Encoder(tensorflow.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tensorflow.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tensorflow.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tensorflow.zeros((self.batch_sz, self.enc_units))

class EDAttention(tensorflow.keras.layers.Layer):
    def __init__(self, units):
        super(EDAttention, self).__init__()
        self.W1 = tensorflow.keras.layers.Dense(units)
        self.W2 = tensorflow.keras.layers.Dense(units)
        self.V = tensorflow.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tensorflow.expand_dims(query, 1)
        score = self.V(tensorflow.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tensorflow.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tensorflow.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class Decoder(tensorflow.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tensorflow.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tensorflow.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")
        self.fc = tensorflow.keras.layers.Dense(vocab_size)
        self.attention = EDAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tensorflow.concat([tensorflow.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tensorflow.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

def loss_function(real, pred):
    mask = tensorflow.math.logical_not(tensorflow.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tensorflow.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tensorflow.reduce_mean(loss_)

def train_step(inp, targ, enc_hidden):
    loss = 0
    with tensorflow.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tensorflow.expand_dims([targ_lang.word_index["<start>"]] * BATCH_SIZE, 1)
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tensorflow.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

def evaluate(sentence):
    attention_plot = numpy.zeros((max_length_targ, max_length_inp))
    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(" ")]
    inputs = tensorflow.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding="post")
    inputs = tensorflow.convert_to_tensor(inputs)
    result = ""
    hidden = [tensorflow.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tensorflow.expand_dims([targ_lang.word_index["<start>"]], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        attention_weights = tensorflow.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()
        predicted_id = tensorflow.argmax(predictions[0]).numpy()
        result += targ_lang.index_word[predicted_id] + " "
        if(targ_lang.index_word[predicted_id] == "<end>"):
            return result, sentence, attention_plot
        dec_input = tensorflow.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot

def plot_attention(attention, sentence, predicted_sentence):
    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap="viridis")

    fontdict ={"fontsize": 14}

    ax.set_xticklabels([""] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([""] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))

    matplotlib.pyplot.show()

def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print("Input: %s" % (sentence))
    print("Predicted translation: {}".format(result))

    attention_plot = attention_plot[:len(result.split(" ")), :len(sentence.split(" "))]
    plot_attention(attention_plot, sentence.split(" "), result.split(" "))

#데이터 호출 및 변수 초기화
num_examples = 118964
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset("./spa.txt", num_examples)
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2, random_state=10)

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1

dataset = tensorflow.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

#인코더, 어텐션, 디코더 및 optimizer, loss 설정
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
attention_layer = EDAttention(10)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

optimizer = tensorflow.keras.optimizers.Adam()
loss_object = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

#체크포인트 설정
checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tensorflow.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

#모델 학습 시작
EPOCHS = 10

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print("Epoch {} Batch {} Loss {:.4f}".format(epoch+1, batch, batch_loss.numpy()))

    if (epoch+1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print("Epoch {} Loss {:.4f}".format(epoch+1, total_loss/steps_per_epoch))

print("Time taken for 1 epoch {} sec\n".format(time.time()-start))

#번역 예시 및 시각화
checkpoint.restore(tensorflow.train.latest_checkpoint(checkpoint_dir))
translate(u"esta es mi vida.")
