# Nanogpt

本项目复现Karpathy的gpt2项目

项目地址：[karpathy/nanoGPT: The simplest, fastest repository for training/finetuning medium-sized GPTs. (github.com)](https://github.com/karpathy/nanoGPT)

b站视频：[【精校】“让我们重现GPT-2（1.24亿参数）!”AI大神Andrej Karpathy最新4小时经典教程 【中英】_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV12s421u7sZ/?spm_id_from=333.1007.top_right_bar_window_custom_collection.content.click&vd_source=48aaa0d1fc3b556c23a01feba661a1ee)

## 1. Before we start

### 1.1 让我们先了解gpt2的框架设计

```python
from transformers import GPT2LMHeadModel
model_hf = GPT2LMHeadModel.from_pretrained("gpt2") #124M
sd_hf = model_hf.state_dict()

for k,v in sd_hf.items():
    print(k,v.shape)
```

```
transformer.wte.weight torch.Size([50257, 768])
transformer.wpe.weight torch.Size([1024, 768])
transformer.h.0.ln_1.weight torch.Size([768])
transformer.h.0.ln_1.bias torch.Size([768])
# ...... #
transformer.h.11.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.11.mlp.c_proj.bias torch.Size([768])
transformer.ln_f.weight torch.Size([768])
transformer.ln_f.bias torch.Size([768])
lm_head.weight torch.Size([50257, 768])
```

- `state_dict() `方法检索模型的状态字典。状态字典是一个 Python 字典对象，它将每个层映射到其参数张量。
- **`transformer.wte.weight:`** weight token embedding 嵌入token层，权重大小50257*768（50257是gpt2的词汇表中token的总数，768是隐藏层维度）
- **`transformer.wpe.weight`:** weight position embedding 位置编码层
- **`transformer.h.0.ln_1.weight`:** 第0个堆叠的隐藏层中的线性层1，gpt2堆叠了12个这样的隐藏层

## 1.2 something about 位置编码

Actually, transformer（Attention is all you need）采用余弦位置编码，但在gpt2中则训练了位置编码transformer.wte.weight。我们可以打印它的内容：

```python
import matplotlib.pyplot as plt
plt.plot(sd_hf["transformer.wpe.weight"][:,150])
plt.plot(sd_hf["transformer.wpe.weight"][:,200])
plt.plot(sd_hf["transformer.wpe.weight"][:,250])
```

![image-20240909095441231](C:\Users\16273\AppData\Roaming\Typora\typora-user-images\image-20240909095441231.png)

- 绘制的图像分别为嵌入维度为150/200/250的位置编码。注意，位置编码是一个[1024,768]的矩阵，因此绘制的曲线实际上分别是一个[1024,1]的向量。它表示了input中1024个token位置的**位置权重**。由于wpe是可学习的，因此768个维度各自对应一组位置权重，且各不相同。Perhaps，它们捕获不同维度独立的位置信息。
- 以绿色曲线为例，该维度的位置编码prefer800-900左右的位置，而接近0的位置和900-1000左右的位置则出现明显衰减。（这提示我们，is position bias in llm come from position embedding？）
- 事实上，我们认为位置编码随着训练的进行尽可能平滑，而gpt2的位置编码显然没有很平滑，说明它可能没有得到充分的训练；此外，我们希望位置编码有正弦/余弦的特性。而gpt2的结果显然不尽如人意。然而，我们仍能在曲线中看到一些类似正弦/余弦的波动。

## 2. Build our GPT2

### 2.1 复现GPT2框架

**注意：**以下内容采用的是视频中为优化的版本，优化的版本为github仓库内版本

1. 首先remember，gpt2对transformer做了改进，删除了encoder结构，保留decoder-only。此外，交叉注意力自然也被删除。以下是transformer框架：

   ![img](https://i-blog.csdnimg.cn/blog_migrate/cfd6415e93225ee678af038e697ee752.png)

此外，gpt2的两个重要改变

- 每个块（灰色框内）中的归一化层位置改变了
- 在最后的分类层前添加了一个层归一化

**实际组成**：

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
```

- `h`为堆叠的隐藏层
- `ln_f`为gpt2新添加的在分类层前的层归一化
- `lm_head`为分类层（即图片中的`Linear`）

**以下为隐藏层Block：**

```python
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

- Here，声明一个基础知识，残差的实现实际上就是x = <u>x</u> + layer(x)

**以下为MLP：**

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
```

- Why use GELU rather than RELU？
  - RELU 在 x<0 时会导致死神经元，而GELU则平滑的给出了一个较小的值

**以下为Attention：**

```python
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y =self.c_proj(y)
        return y
```

**配置：**

```python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12 # numbers of attention heads
    n_embd: int = 768
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
```

**前向传播：**

```python
def forward(self, idx):
        B, T = idx.size()
        assert T<= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
```

- 输入为token的index，形状为B*T。B是batch，T是tokens数（最大为1024，即模型支持的上下文长度）

**using model：**

```py
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'using device: {device}')        

num_return_sequences = 5
max_length = 30

model = GPT(GPTConfig())
model.eval()
model.to(device)

# 编码
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # 使用torch.no_grad()上下文管理器,在推理过程中禁用梯度计算
    with torch.no_grad():
        # 将当前序列输入模型,获取输出logits
        logits = model(x)
        # 只选择最后一个时间步的输出logits
        logits = logits[:, -1, :]
        # 使用softmax函数将logits转换为概率分布
        probs = F.softmax(logits, dim=-1)
        # 选择概率最高的前50个token
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # 从这50个token中按概率随机采样一个
        ix = torch.multinomial(topk_probs, 1)
        # 获取被采样token的实际索引
        xcol = torch.gather(topk_indices, -1, ix)
        # 将新生成的token添加到现有序列的末尾
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
```

### 2.2 数据集构建

**以shakespeare为例**

#### 2.2.1 tiny example

1. 加载数据

```py
with open('data/shakespeare_char/input.txt', 'r') as f:
    text = f.read()
data = text[:1000]
print(data[:100])
```

```
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You
```

2. 编码

```py
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode(data)
print(tokens[:24])
```

```
[5962, 22307, 25, 198, 8421, 356, 5120, 597, 2252, 11, 3285, 502, 2740, 13, 198, 198, 3237, 25, 198, 5248, 461, 11, 2740, 13]
```

3. 转换为B*T张量

```
import torch 
buf = torch.tensor(tokens[:24])
x = buf[:].view(4,6)
print(x)
```

```
tensor([[ 5962, 22307,    25,   198,  8421,   356],
        [ 5120,   597,  2252,    11,  3285,   502],
        [ 2740,    13,   198,   198,  3237,    25],
        [  198,  5248,   461,    11,  2740,    13]])
```

however, 我们需要根据x预测下一个token，但很显然x中最后一个token（13）的下一个token是未知的，因此采用以下的方法构造数据集

4. 优化

```
import torch 
buf = torch.tensor(tokens[:24+1])
x = buf[:-1].view(4,6)
y = buf[1:].view(4,6)
print(x)
print(y)
```

```
tensor([[ 5962, 22307,    25,   198,  8421,   356],
        [ 5120,   597,  2252,    11,  3285,   502],
        [ 2740,    13,   198,   198,  3237,    25],
        [  198,  5248,   461,    11,  2740,    13]])
tensor([[22307,    25,   198,  8421,   356,  5120],
        [  597,  2252,    11,  3285,   502,  2740],
        [   13,   198,   198,  3237,    25,   198],
        [ 5248,   461,    11,  2740,    13,   198]])
```

- 其中，x是输入，y是target

#### 2.2.2 数据读取

```py
import tiktoken
enc = tiktoken.get_encoding('gpt2')
with open('data/shakespeare_char/input.txt', 'r') as f:
    text = f.read()
data = text[:1000]
tokens = enc.encode(text)
B, T = 4, 32
buf = torch.tensor(tokens[:B*T+1])
x = buf[:-1].view(B,T)
y = buf[1:].view(B,T)
```

### 2.3 构造训练代码

#### 2.3.1 loss 

```
# model = GPT(GPTConfig())
# model.to(device)
# logits = model(x)

model = GPT(GPTConfig())
model.to(device)
logits, loss = model(x)
```

修改forward函数，在返回值中添加loss

```
def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T<= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
```

- targets即2.2.2中的`y`，表示预测的目标

- `loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))`中，由于`F.cross_entropy`不接收三维张量，因此将`logits`展平成2维，`targets`展平成1维

  - `logits.view(-1, logits.size(-1))`

    - `logits` 通常是一个三维张量，形状为 (batch_size, sequence_length, vocab_size)

    - `logits.size(-1)` 获取最后一个维度的大小，通常是词汇表的大小（vocab_size）

    - ```py
      view(-1, logits.size(-1))
      ```

      将 logits 重塑为二维张量：

      - `-1` 自动计算这个维度的大小，使其与原始数据的总元素数量匹配
      - 结果形状为 (batch_size * sequence_length, vocab_size)

  - `targets.view(-1)`

    - `targets` 通常是一个二维张量，形状为 (batch_size, sequence_length)
    - `view(-1)` 将其展平为一维张量
    - 结果形状为 (batch_size * sequence_length)

**\* 如何检查loss是否正确**

- 在模型初始化时，我们希望每一个token（for gpt2, it's 50257）的初始预测概论是尽可能相等的，这是一个好的无偏好的起点。
- 交叉熵损失基本上是负对数似然
- `ln(1/50257)=-10.8249`
- `print(loss)`,得到结果`tensor(10.9575, grad_fn=<NllLossBackward0>)`，可见与估计值基本吻合

#### 2.3.2 优化和反向传播

```py
# logits, loss = model(x, y)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")
```

- torch默认会梯度累积，因此需要手动zero_grad

**Dataloader**

```py
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('data/shakespeare_char/input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)
        self.current_position +=B*T

        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0
        return x, y
```

循环更新为：

```py
train_loader = DataLoaderLite(B=4, T=32)

model = GPT(GPTConfig())
model.to(device)
# logits, loss = model(x, y)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")
```

#### 2.3.3 fix a bug: 权重共享

- 在attention is all you need中3.4节提到，`In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear trasnformation, similar to [30].`在gpt2中也采用了同样的设计，即`transformer.wte`和`lm_head`共享完全相同的权重矩阵。

- 这意味着，在反向传播时，`transformer.wte`的权重获得的梯度来自于两条分支的贡献：①来自于模型的top，即`lm_head`的贡献②来自于模型的bottom，即`transformer.wte`的贡献

- 实现

  ```python
  class GPT(nn.Module):
      def __init__(self, config):
          super().__init__()
          self.config = config
          self.transformer = nn.ModuleDict(dict(
              wte = nn.Embedding(config.vocab_size, config.n_embd),
              wpe = nn.Embedding(config.block_size, config.n_embd),
              h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
              ln_f = nn.LayerNorm(config.n_embd),
          ))
          self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
          
          # weight sharing scheme
          self.transformer.wte.weight = self.lm_head.weight
  ```

- **解释**：从维度上看，似乎两个权重矩阵是转置关系，为什么代码没有体现？

  - 实际原因是，torch对于Embedding层和Linear层的实现不同，
    - nn.Embedding：$XW$
    - nn.Linear：$XW^T + b$
  - 因此，线性层本身的权重就天然带有转置。

- 权重共享的另一个好处是减少计算

  - transformer的参数中token embedding占据很大的部分
    - 768*50257≈4000万
    - gpt2本版本参数量共1.24亿
    - token embedding占据约30%的参数

#### 2.3.4 权重初始化

- 权重初始值决定了模型训练的起点。有一些理论性工作探讨。除此以外似乎更多的是工程实践决定的。gpt2给出了Linear、Embedding层的权重初始化设计，尽管没有给出原因。为了复现gpt2，我们需要添加权重初始化代码

```python
class GPT(nn.Module):
    def __init__(self, config):
        # ... 
        
        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

- `apply`函数：迭代此模块的所有子模块，并对它们应用`_init_weights`函数

- `isinstance()` 函数: 

  - 语法: `isinstance(object, classinfo)`

  - 作用: 检查一个对象是否是一个类或者元组中任意类的实例。

  - 参数:

    - `object`: 要检查的对象

    - `classinfo`: 类或者由类组成的元组

- 一般而言，std应该设置为1/d，其中d是隐藏层维度数。然而，在这里将其硬编码为0.02基本是影响不大的。因为
  - 1/768≈0.03
  - 其它接近这个数量级的隐藏层维度数也类似

#### 2.3.5 抑制残差累计

- 在gpt2报告中提到，他们使用了一个修改后的初始化，考虑了模型深度在残差路径上的累积。将残差层初始化的权重按$1/\sqrt(N)$,其中$N$是残差层的数量。

- 原理：对于包含多个残差的网络而言，残差流中激活的方差会累积。当网络变深时，这种方差累积会导致后续层的输入方差不断增大。

  - 在残差网络中，每一层的输出 y 可以表示为：`y = F(x) + x`，其中` x `是输入，`F(x) `是残差函数。
  - 假设输入` x `的方差为 `Var(x)`，残差函数 `F(x)` 的输出方差为 `Var(F(x))`。如果 `x `和 `F(x) `是独立的，那么输出 `y `的方差将是：`Var(y) = Var(F(x)) + Var(x)`
  - 当网络变深时，这种方差累积会导致后续层的输入方差不断增大。例如，经过 `n `个残差块后，假设每个残差函数的方差相同，输出的方差可能变为：`Var(y_n) ≈ n * Var(F(x)) + Var(x_0)`
  - 这可能导致激活值的幅度可能会变得非常大。大的激活值可能导致梯度爆炸或消失。
  - **解决方法**：引入一个小于 1 的缩放因子 `α`：`y = αF(x) + x`。在gpt2的实现中，实际上是在参数初始化时减小残差流相关层的初始方差。
  - **补充说明**：事实上在非残差网络中类似的方差累计现象也可能存在，主要与权重矩阵`W`的二范数有关。`W`的二范数大于1则造成方差累加，小于1则造成方差衰减。因此在层之后进行归一化是有必要的。

- 实现：在权重初始化时，对MLP和Attention中涉及到残差计算的层，降低其初始的std，从而减小残差块对主路径的初始贡献

  ```python
  class CausalSelfAttention(nn.Module):
      def __init__(self, config):
          # ...
          self.c_proj = nn.Linear(config.n_embd, config.n_embd)
          self.c_proj.NANOGPT_SCALE_INIT = 1
  ```

  ```py
  class MLP(nn.Module):
      def __init__(self, config):
  		# ...
          self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
          self.c_proj.NANOGPT_SCALE_INIT = 1
  ```

  ```py
      def _init_weights(self, module):
          if isinstance(modeule, nn.Linear):
              std = 0.02
              if hasattr(module, 'NANOGPT_SCALE_INIT'):
                  std *= (2 * self.config.n_layer)  ** -0.5
              torch.nn.init.normal_(module.weight, mean=0.0, std=std)
              if module.bias is not None:
                  torch.nn.init.zeros_(module.bias)
          elif isinstance(module, nn.Embedding):
              torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
  ```

- 为什么是这两个层？从Block函数的定义可以看出来，这两个层的输出分别构成了残差流

- 为什么是`2 * self.config.n_layer`？因为一个Block包含`2`个残差流（1个MLP，1个Attention）

- `hasattr()` 函数:

  - 语法: `hasattr(object, name)`

  - 作用: 检查一个对象是否具有指定的属性。

  - 参数:

    - `object`: 要检查的对象

    - `name`: 要检查的属性名称（字符串）

#### 3.3.6 随机种子

```
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
```

### 2.4 优化训练速度

- 为了优化训练速度，首先先确定当前代码的速度作为baseline

  B = 16, T=1024

  ```py
  import time
  for i in range(50):
      t0 = time.time()
      x, y = train_loader.next_batch()
      x, y = x.to(device), y.to(device)
      optimizer.zero_grad()
      logits, loss = model(x, y)
      loss.backward()
      optimizer.step()
      torch.cuda.synchronize() # 等待gpu完成所有工作
      t1 = time.time()
      dt = (t1-t0)*1000
      print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms")
  ```

  ```
  step 49, loss: 6.01735782623291, dt: 807.30ms
  ```

  时间在800ms/epoch左右

#### 2.4.1 TensorFloat32

- 只需要添加一行代码实现TensorFloat32精度。所有变量仍然是float32，但是运行速度更快，更近似。

```
torch.set_float32_matmul_precision('high')
```

- 这可能只适用于A系列显卡，较老的显卡可能不支持（A6000亲测可行）

- 效果

  ```
  step 49, loss: 6.017364978790283, dt: 539.33ms
  ```

  优化了30%左右

#### 2.4.2 BFloat16

- 只需要添加一行代码实现BFloat16精度。它可以与TensorFloat32一起使用，因为并不是所有的计算都会使用BFloat16
  - 例如softmax、normalize等易受精度变化影响的运算仍旧会使用TensorFloat32

```py
for i in range(50):
    # ...
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16): # 添加这一行
        logits, loss = model(x, y)
```

- 这可能只适用于A系列显卡，较老的显卡可能不支持（A6000亲测可行）

- 效果

  ```
  step 49, loss: 6.018825531005859, dt: 414.78ms
  ```

  进一步优化了20%左右

#### 2.4.3 torch.compile

- 神经网络的编译器。它会花费额外的编译时间，但是运行速度大大加快。鼓励默认使用`torch.compile`，除非是在代码调试过程中

  ```py
  model = GPT(GPTConfig())
  model.to(device)
  model = torch.compile(model)
  ```

- 效果

  ```
  step 49, loss: 6.019706726074219, dt: 226.40ms
  ```

  进一步优化了50%左右

- 它可以减少GPU读写的时间。

  - 因为GPU在芯片上的存储空间是很小的，他需要与GPU中的HBM交互，将读写数据到HBM中。总而言之，芯片上的读写速度极快但存储空间极小，HBM提供了GPU的内部存储空间，但读写速度约是前者的10%。
  - torch.compile自动检测代码中的逻辑。当一份数据被读到芯片上后，正常来讲在计算结束后该数据就被写回HBM。然而torch.compile会分析一段代码中这份数据是否会被重复利用，若会，则将其保留在芯片上，等待计算结束再写回HBM。这节省了很多读写时间。

#### 2.4.4 Flash Attention

- Faster。因为它更加关注内存层次结构，减少读写次数。

- 修改方式：将Attention中以下四行改为这一行代码

  ```py
  # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
  # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
  # att = F.softmax(att, dim=-1)
  # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
  y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
  ```

- 效果

  ```
  step 49, loss: 6.0193634033203125, dt: 151.45ms
  ```

  进一步优化了30%左右

#### 2.4.5 数值设置

- 事实上，使用gpu训练的神经网络中有nice数字和ugly数字。$2^n$通常是nice数字，例如64、128。奇数和质数通常是ugly数字，例如13、17。

- 首先我们要知道的是，在gpu中张量乘法运算基本上都会被分解维4*4矩阵乘法运算。

- cuda的许多运算都是以2的幂形式工作，许多内核都以2的幂形式编写。因此，使用奇数或质数会导致需要对各种逻辑进行特殊处理，例如额外的启动来完成奇数所剩余的最后这一部分计算。

- 修改：例如，vocab_size: int = 50257是一个ugly数字，因为它不是2的倍数。我们可以将它修改为50304。（这里我要道歉，在上述代码中我实际上是以50304进行试验，正常来讲到这一步才会优化。不过我会在此小节单独对比修改前后的改进）

  ```py
  @dataclass
  class GPTConfig:
      block_size: int = 1024
      vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
  ```

- 效果

  **50257**

  ```
  step 49, loss: 6.056464672088623, dt: 164.09ms
  ```

  **50304**

  ```
  step 49, loss: 6.0193634033203125, dt: 151.45ms
  ```

  优化了8%左右

## 3. 进一步优化

### 3.1 像GPT3一样设置参数

#### 3.1.1 设置Adam参数

```py
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
```

#### 3.1.2 梯度裁剪

在反向传播后添加以下这一行

```
loss.backward()
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

- 该函数计算所有参数梯度的总范数(L2范数),并将其缩放,使得总范数不超过指定的最大值(这里是1.0)。函数返回裁剪前的原始梯度范数,这里赋值给了`norm`变量。
- 作用：防止梯度爆炸
- Karpathy说，他喜欢将clip_grad_norm_返回的范数可视化，因为它是有用的信息。**如果梯度的范数不断攀升，模型训练就很糟糕，可能导致不稳定。如果看到范数出现一个峰值，可能表示出现了不稳定的情况。**我们在print处加一个norm指标来打印范数

#### 3.1.3 学习率

gpt训练采用的学习率是从余弦函数改造的，大体如下图所示：

<img src="C:\Users\16273\AppData\Roaming\Typora\typora-user-images\image-20240912164352028.png" alt="image-20240912164352028" style="zoom:80%;" />

**学习率函数：**

```py
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1)/warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min leaning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
```

- 1) 预热阶段（Warm-up）：在训练开始的 `warmup_steps` 步内，学习率从接近0线性增加到 `max_lr`。这有助于稳定初期的训练过程。
  2) 最小学习率阶段：当迭代次数超过 `max_steps` 时，返回最小学习率 `min_lr`。
  3) 余弦退火阶段（Cosine Annealing）：在预热阶段之后到 `max_steps` 之前，学习率按余弦函数从 `max_lr` 逐渐降低到 `min_lr`。

**训练循环：**

```py
for step in range(max_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize() # 等待gpu完成所有工作
    t1 = time.time()
    dt = t1-t0
    print(f"step {i}, loss: {loss.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms")
```

- 说明：在gpt3中，max_steps是比训练伦次要少的，也就是说训练中存在epoch（在较后期）以min_lr训练。但在我们的代码中，训练轮次设置为max_steps，表示训练中没有以min_lr为学习率的epoch。

#### 3.1.4  growing Batch

- gpt3逐步增加了batch大小。batch随着训练轮次的增加线性上升，开始以一个小batch训练，渐渐增加到一个大的batch
- 这似乎是合理的，解释有很多（不仅限于以下解释）
  - warmup阶段采用小batch可以频繁更新参数，有助于快速探索参数空间
  - 小batch在早期阶段提供更多的随机性，有助于模型学习数据的多样性
  - 小batch在训练初期提供更多的噪声，有助于逃离局部最优解，增加模型的泛化能力

- 本文并不复现这一操作。因为这很麻烦。

#### 3.1.5 权重衰减

权重衰减是正则化，避免单个权重非常大。

修改优化器为自定义的优化器：

```
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
```

在模型内部添加一个优化器函数：

```py
import inspect

class GPT(nn.Module):
    # ...

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
```

- 一维向量不参与权重衰减，我们只对参与矩阵乘法的权重进行衰减，以及对embedding的权重衰减

#### 3.1.6 梯度累积

- 当我们想以一个很大的batch训练时，显存往往是不够的。gpt3使用0.5M个toens为一个batch，这需要我们将B设置为488左右。这在一个gpu上是几乎实现不了的。

- 梯度累积指的是，在反向传播后不要直接更新参数，而是将梯度保留。在运行若干个迷你batch后，梯度会累加，最终一次性更新参数。

- 实现：

  ```py
  # train_loader = DataLoaderLite(B=16, T=1024)
  total_batch_size = 524288 # 2^19 ≈0.5M
  B = 16 # micro batch
  T = 1024 # sequence length
  assert total_batch_size % (B*T) == 0, "make sure total_batch_size is divisible by B*T"
  grad_accum_steps = total_batch_size // (B * T)
  print(f"total desired batch size: {total_batch_size}")
  print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
  
  train_loader = DataLoaderLite(B=B, T=T)
  ```

  ```py
  for step in range(max_steps):
      t0 = time.time()
      optimizer.zero_grad()
  
      loss_accum = 0.0
      for micro_step in range(grad_accum_steps):
          x, y = train_loader.next_batch()
          x, y = x.to(device), y.to(device)
          with torch.autocast(device_type=device, dtype=torch.bfloat16):
          	logits, loss = model(x, y)
  
          loss = loss / grad_accum_steps
          loss_accum += loss.detach()
          loss.backward()
      norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      
      lr = get_lr(step)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  
      optimizer.step()
      torch.cuda.synchronize() # 等待gpu完成所有工作
      t1 = time.time()
      dt = t1-t0
      print(f"step {step}, loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms")
  ```

  - 注意：`loss = loss / grad_accum_steps`表示，在梯度累积时需要除以执行micro_step的次数。因为在执行1个n批次和n个1批次时，区别在于执行n个1批次需要累加梯度，但次累加并不会默认归一化，这就会导致与执行1个n批次的损失结果**相差n倍**
  - `optimizer.step()`更新参数。因此只有一个epoch才更新一次参数，该次更新累积了grad_accum_steps次梯度

### 3.2 分布式多卡训练 DDP

指令：`torchrun --standalone --nproc_per_node=4 train_gpt2.py`

#### 3.2.1 初始化ddp

```py
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f'using device: {device}') 

# DDP
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl') # 初始化进程组，使用NCCL后端
    ddp_rank = int(os.environ['RANK']) # 获取当前进程的全局rank。在分布式环境中，每个进程都有一个唯一的rank。
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # 当前进程（GPU）在本地机器上的rank
    ddp_world_size = int(os.environ['WORLD_SIZE']) # 总进程数（世界大小）
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # 主进程，用于打印一些内容等
else:
    ddp_rank = 0 
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'using device: {device}')  
```

#### 3.2.2 修改训练代码

1. micro batch分配

   ```py
   assert total_batch_size % (B*T*ddp_world_size) == 0, "make sure total_batch_size is divisible by B*T*ddp_world_size"
   grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
   if master_process:
       print(f"total desired batch size: {total_batch_size}")
       print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
   ```

2. 数据加载

   ```py
   # train_loader = DataLoaderLite(B=B, T=T)
   train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)
   ```

   ```python
   class DataLoaderLite:
       def __init__(self, B, T, process_rank, num_processes):
           self.B = B
           self.T = T
           self.process_rank = process_rank
           self.num_processes = num_processes
   
           with open('data/shakespeare_char/input.txt', 'r') as f:
               text = f.read()
           enc = tiktoken.get_encoding('gpt2')
           tokens = enc.encode(text)
           self.tokens = torch.tensor(tokens)
           print(f"loaded {len(self.tokens)} tokens")
   
           self.current_position = self.B * self.T * self.process_rank
   
       def next_batch(self):
           B, T = self.B, self.T
           buf = self.tokens[self.current_position : self.current_position+B*T+1]
           x = (buf[:-1]).view(B,T)
           y = (buf[1:]).view(B,T)
           self.current_position +=B*T*self.num_processes
   
           if self.current_position + (B*T*self.num_processes+1) > len(self.tokens):
               self.current_position = self.B * self.T * self.process_rank
           return x, y
   ```

   - 注意：每个进程从不同的起始位置开始读取数据：`self.current_position = self.B * self.T * self.process_rank`
     - 从 `current_position` 开始，读取 `B*T+1` 个tokens。
     - 更新 `current_position`，跳过其他进程的数据：`self.current_position += B*T*self.num_processes`

3. 将模型分装到分布式容器

   ```py
   model = GPT(GPTConfig())
   model.to(device)
   model = torch.compile(model)
   if ddp:
       model = DDP(model, device_ids=[ddp_local_rank])
   ```

   - 在最简单的设置中，一旦每个独立GPU上的反向传播结束，每个GPU都会拥有所有参数的梯度。DDP在这里做的是：一旦反向传播结束，它会调用`all_reduce`，**对所有梯度求平均值，然后将该平均值存储在每个GPU（Rank）上，因此每个Rank都会得到平均值**。

4. 反向传播同步时机优化

   在`for step in range(max_steps):`循环中修改：

   ```py
   if ddp:
   	model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
   loss.backward()
   ```

   - DDP默认会在反向传播后同步各个Rank（GPU）上的参数梯度。但由于我们使用了梯度累积，我们其实只希望在最后一次micro step结束后同步梯度，因为这样会减少不必要的通信损耗。
   - 这段代码让DDP只有在micro_step == grad_accum_steps - 1时才同步梯度。

5. loss同步

   ```py
   if ddp:
   	dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
   norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   ```

   - 与反向传播不同，DDP中并不会自动同步loss，因此需要我们手动处理，调用`all_reduce`计算所有Rank（GPU）上loss的平均值并将它们存储到每一个Rank上

6. 打印主进程结果

   ```py
   if master_process:
   	print(f"step {step}, loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms")
   ```

7. 销毁进程组

   将这段代码添加到最后（训练循环的循环外面）

   ```python
   if ddp:
       destroy_process_group()
   ```

8. 修改一些bug

   1. **BUG 1**

      ```py
      raw_model = model  # 保存原始模型的引用
      if ddp:
          model = DDP(model, device_ids=[ddp_local_rank])
      ```

      ```
      optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
      ```

      - 这里是因为DDP在包装model时，导致了model内的函数configure_optimizers没有被暴露出来。因此直接用`model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)`会报错找不到`configure_optimizers`函数。因此我们保存了原始的model，并在此处调用原始model的引用，而不是DDP包装的model。

   2. **BUG 2**

      ```py
      device_type = device.split(':')[0]  # 这会从 'cuda:0' 中提取 'cuda'
      with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
      	logits, loss = model(x, y)
      ```

      - 我没有意识到视频什么时候修改了这里。出错原因是autocast函数只接收'cuda'/'cpu'，而我们的device是'cuda: {id}',例如'cuda: 0'