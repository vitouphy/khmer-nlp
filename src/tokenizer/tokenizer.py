import numpy as np
from .network import Network
from .utils import read_pos_map, read_char_map, read_config

config = read_config('src/tokenizer/configs/original_paper.json')
char_map = read_char_map('src/tokenizer/pretrained/char_map.txt')
pos_map = read_pos_map('src/tokenizer/pretrained/pos_map.txt')
weight_path = 'src/tokenizer/pretrained/original_paper.weights.h5'

num_chars = len(char_map)
pos_to_index = {pos: i for i, pos in enumerate(pos_map)}
index_to_pos = {i: pos for i, pos in enumerate(pos_map)}
char_to_index = {char: i for i, char in enumerate(char_map)}

model = Network(
    output_dim=len(pos_map),
    embedding_dim=len(char_map),
    num_stacks=config["model"]["num_stacks"],
    hidden_layers_dim=config["model"]["hidden_layers_dim"],
    max_sentence_length=config["model"]["max_sentence_length"],
)
model.load_weights(weight_path, by_name=True)

def preprocess_sample(sentence):
    sentence_input_vector = np.zeros((config["model"]["max_sentence_length"], num_chars))
    for i, char in enumerate(sentence):
        if char in char_to_index:
            char_index = char_to_index[char]
        else:
            char_index = char_to_index["UNK"]
        sentence_input_vector[i, char_index] = 1

    return sentence_input_vector, sentence

def tokenize(samples):
    result = []
    for sample in samples:
        sentence_input_vector, sentence = preprocess_sample(sample)
        pred = model.predict(np.array([sentence_input_vector]))[0]

        words, tmp, pos = [], [], []
        for char_idx, pos_vector in enumerate(pred):
            if char_idx < len(sentence):
                pos_index_pred = np.argmax(pos_vector)

                if index_to_pos[pos_index_pred] == "NS":
                    tmp.append(sentence[char_idx])
                else:
                    if len(tmp) > 0:
                        words.append("".join(tmp))
                        tmp = []
                    pos.append(index_to_pos[pos_index_pred])
                    tmp.append(sentence[char_idx])

        if len(tmp) > 0:
            words.append("".join(tmp))
        result.append((words, pos))
        
    return result

