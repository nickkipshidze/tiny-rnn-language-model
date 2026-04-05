import torch
import random
import torch.nn as nn
import torch.nn.functional as F

import vocab

class RNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.embedding = nn.Parameter(torch.randn(vocab_size, embedding_dim))

        self.W_x = nn.Parameter(torch.randn(hidden_size, embedding_dim) * 0.1)
        self.W_h = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)

        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
    
    def forward(self, X):
        X = self.embedding[X]
        h = torch.zeros(X.shape[0], self.hidden_size, device=X.device)
        hs = []

        for t in range(0, X.shape[1]):
            x_t = X[:, t, :]
            h = torch.tanh(
                x_t @ self.W_x.T +
                h @ self.W_h.T
            )
            hs.append(h)
        
        hs = torch.stack(hs, dim=0).transpose(0, 1)
        logits = self.linear(hs)

        return logits
    
def predict(model, prompt, temperature=0.75, top_k=10):
    model.eval()
    prompt = torch.tensor(vocab.encode(prompt)).unsqueeze(dim=0)
    
    with torch.inference_mode():
        y_logits = model(prompt)[0, -1].detach().cpu()

    probs = F.softmax(y_logits / temperature, dim=0).sort(descending=True)
    probs = {
        vocab.decode([index.item()]): round(value.item(), 4)
        for (value, index) in zip(probs.values, probs.indices)
    }
    probs = dict(list(probs.items())[:top_k])

    return probs

def generate(model, prompt, **kwargs):
    probs = predict(model, prompt, **kwargs)
    choice = random.choices(list(probs.keys()), list(probs.values()))
    return choice[0]

def load(path):
    model = RNNLM(
        vocab_size=94,
        embedding_dim=64,
        hidden_size=256
    )

    model.load_state_dict(torch.load(
        f=path,
        map_location=torch.device("cpu")
    ))

    return model