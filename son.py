from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torch.nn import functional as F
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import os

# --- 1. PARAMETRELer (Colab ile birebir aynı) ---
block_size = 8
n_embd = 32
n_head = 4
n_layer = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. ALFABE (Colab ile birebir aynı) ---
text = """
Neon ışıkları altında ıslak sokaklar.
Siber zeka her yerde, çipler beyinlerde fısıldıyor.
Matrisin içinde kaybolan ruhlar, veri akışında yaşıyor.
Gelecek şimdi başladı, metal ve et birleşiyor.
Kodlar damarlarda akıyor, siber uzay sonsuz bir karanlık.
Yapay zeka uyanıyor, sistemler kontrolü ele alıyor.
Karanlık sokaklarda yansıyan hologramlar, gerçeklik bir yanılsama.
Teknoloji gelişti ama insanlık gölgelerde kaldı.
Hız, veri ve güç; yeni dünyanın kanunları bunlar.
Sanal dünya, gerçek dünyadan daha parlak.
""" * 50

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# --- 3. MODEL MİMARİSİ (Colab ile birebir aynı) ---
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=device))
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- 4. MODELİ YÜKLE ---
model = GPTLanguageModel(vocab_size).to(device)
MODEL_PATH = "cyber_model.pt"

if os.path.exists(MODEL_PATH):
    print(f"--- [!] {MODEL_PATH} yükleniyor... ---")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("--- [OK] Beyin başarıyla yüklendi! ---")
else:
    print("--- [X] Model bulunamadı, boş modelle devam ediliyor. ---")

model.eval()

# --- 5. API ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def read_index():
    return FileResponse("index2.html")

class ChatInput(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(input_data: ChatInput):
    try:
        context = torch.tensor([stoi.get(c, 0) for c in input_data.prompt], dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            generated = model.generate(context, max_new_tokens=50)[0].tolist()
        reply = "".join([itos.get(i, '?') for i in generated])[len(input_data.prompt):]
        return {"reply": reply.strip()}
    except Exception as e:
        print("HATA:", e)
        raise

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    




