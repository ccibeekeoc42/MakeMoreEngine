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
With the Bigram model, we predict the next character in the sequence using a simple lookup table containing bigram counts. We do this by looking at only two characters at a time. Given one character, we try to predict the next likely character. This is achieved by using a 2-D array where each row represents the first character and each column is the second character. This means each entry in the array is the count of how often the second character follows the first in each sequence. This is a very simple and weak language model.

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

We proceed to create a probability distribution vector for each row in our lookup table (2-D array) and sample items (characters) based off the distribution density. In the code below, we are doing this to generate only 5 words.

```python
# Creating a propability distibution vectors (for each row)
P = (N+1).float()
P = P / P.sum(dim=1, keepdim=True)

g = torch.Generator().manual_seed(2147483647) # generator to ensure deterministic values

for i in range(5):
  out = []
  ix = 0
  while True:
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item() # Drawing 1 sample based off the distribution
    out.append(itos[ix])
    if ix == 0: # breaks if we ever get back to the . character
      break
  print(''.join(out))
```
Below are the 5 words generated. We can observe that this is a terrible sample/performance using just a plain lookup frequency table/ bigram tarining model.

```
mor.
axx.
minaymoryles.
kondlaisah.
anchshizarie.
```

Next we evaluate the performance of this probability distribution Bigram model by calculating the loss. The loss is a single number that tells us how good the model is at predicting based off the training. A good way to achieving this single loss number is know in statistics as the `Maximum Likelihood` and this is just the product of all the probabilities in our probability density table.

For simplicity and convinence, we would be working with the log-likelihood which is the sum of the log of all probabilities in the probability density table. Keep in mind that log is a monotonic transformation where $log(1) \approx 0$ and $log(0) \approx - \infty$. In this case, since probabilities only take on values from $0 - 1$, if we take the log of a probability close to 1, we get something close to 0. Conversely, as we go lower in probability closer to 0, we get more negative and closer to $-\infty$.

Since we are trying to measure the loss/ performance of our model, it is better to use the negative log-likelihood (NLL) instead as this signifies low is good and high (loss) is bad. See the image below. the NLL is just the negative of the log likelihood. The NLL can also be normalized for further simplicity.

<p align="center">
 <img
  src="neg_log.png"
  alt="Computational graph"
  title="Optional title"
  style="display: inline-block; align: center; margin: 0 auto; width:250px">
</p>


Below is the loss calculation code.

```python
# Calculating the loss
log_likelihood = 0
n = 0
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1, ix2 = stoi[ch1], stoi[ch2]
    prob = P[ix1, ix2]
    logProb = torch.log(prob)
    log_likelihood += logProb
    n += 1
    #print(f'{ch1}{ch2}: {prob:.4f} {logProb:.4f}')
print(f'Log Likelihood: {log_likelihood:.4f}')
nnl = -log_likelihood
print(f'NNL: {nnl:.4f}')
print(f'Normalized NNL: {nnl/n:.4f}')
```

### Glossary
- [**Autoregressive Model**](https://www.google.com/search?q=auto+regressive+meaning): A statistical model thaqt predicts future values based on past values.
- [**Maximum Likelihood**](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation): A method of estimating the parameters of a probability distribution, given some observed data.
- [**Probability Distribution**](https://www.google.com/search?q=probability+distribution): A mathematical function that describes the probability of different possible values of a variable.