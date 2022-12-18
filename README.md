# MakeMoreEngine
In this repo, we explore an introduction to language modeling by building several language models. Please Enjoy!

### Software, Tools, and prerequisits
1. Access to Google Colab or some Jupyter Notebook.
2. Basic python programing.
3. Basic PyTorch knowledge.

### Intro: What is MakeMore Engine
As the name suggests, makemore is an engine that makes more of things given to it. For example, if trained on a list of names, MakeMore Engine would generate more unique name-like words based off what it learns from the training dataset (list of names provided). 

MakeMore is a character-level language model that treats each of its training examples as sequences of individual characters. This measn that given some characters in the sequence, it learns to predict the next character.

We would explore several autoregressive models from Bigrams to Transformers (like GPT) of this character level predictions namely:

- [A Bag of Words](https://github.com/ccibeekeoc42/MakeMoreEngine#a-bag-of-Words-bigrams)


### A Bag of Words (Bigrams)
With the Bigram model, we predict the next character in the sequence using a simple lookup table containing bigram counts. We do this by looking at only two characters at a time. Given one character, we try to predict the next likely character. This is achieved with the lookup table of counts. These counts tells us the frequency of how each preceeding character relates to the next character in the sequence. This is a very simple and weak language model.

First we load our dataset which in this case is a list of peoples names. Then we create a bigram lookup table using the character `.` to denote both the end and the start of each sequence (name) in our dataset.

We begin with our basic imports.

```python
import torch
import matplotlib.pyplot as plt
%matplotlib inline
```
Then we proceed to create our bigram lookup table containing frequency of each character pairs.

```python
words = open('names.txt', 'r').read().splitlines()
# Creating lookup table
N = torch.zeros((27, 27), dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1, ix2 = stoi[ch1], stoi[ch2]
    N[ix1, ix2] += 1
```

The table created above can then be viewed using the piece of code below.

```python
plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
  for j in range(27):
    chstr = itos[i] + itos[j]
    plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
    plt.text(j, i, N[i,j].item(), ha='center', va='top', color='gray')
```
<p align="center">
 <img
  src="lookup.png"
  alt="Computational graph"
  title="Optional title"
  style="display: inline-block; align: center; margin: 0 auto;">
</p>
### Glossary
- [**Autoregressive Model**](https://www.google.com/search?q=auto+regressive+meaning): A statistical model thaqt predicts future values based on past values.