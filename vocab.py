vocab = ["\n", " ", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "=", "?", "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "^", "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~"]

char_to_idx = {
    v: i for (i, v) in enumerate(vocab)
}

idx_to_char = {
    i: v for (i, v) in enumerate(vocab)
}

def encode(string):
    return [char_to_idx.get(s, 2) for s in string]

def decode(indices):
    return "".join([idx_to_char.get(i, "!") for i in indices])
