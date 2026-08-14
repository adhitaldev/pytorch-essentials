# Illustration of data flow through the basic transformer

Lets take the following simple sentence:
"The cat sat"
Here are the steps this goes through during the process of training the transformer.

## Step 1 Tokenization and config setup
```
Vocabulary : {"The": 0, "cat": 1, "sat": 2}
vocab_size = 3
n_embd = 4 # defines as dmodel in the paper
head_size = 2
block_size = 4 # block size is a hyperparameter, generally larger than vocab size, and it determines size of position embedding table, and tril mask. It limits how far back the 
model can look at once.
```

## Step 2 Token embedding lookup 
```
tok_emb (vocab_size X n_embd) = 
id0 "The" -> [1.0, 0.0, 1.0, 0.0
id1 "cat" ->  0.0, 1.0, 0.0, 1.0
id2 "sat" ->  1.0, 1.0, 0.0, 0.0]
```

## Step 3 Position embedding loopup
```
pos_emb (block_size X n_embd) = 
pos0 -> [0.1, 0.0, 0.0, 0.1
pos1 ->  0.0, 0.1, 0.1, 0.0
pos2 ->  0.1, 0.1, 0.0, 0.1
pos3 ->  0.1, 0.0, 0.1, 0.1]
```

## Step 4 Addition of token and position embedding
The current number of tokens or the length of this batch
of sequences is __T__. __block_size__ is the maximum __T__ can be.
```
x (T x C)= tok_emb + pos_emb
"The" (pos0): [1.0,0.0,1.0,0.0] + [0.1,0.0,0.0,0.1] = [1.1, 0.0, 1.0, 0.1
"cat" (pos1): [0.0,1.0,0.0,1.0] + [0.0,0.1,0.1,0.0] =  0.0, 1.1, 0.1, 1.0
"sat" (pos2): [1.0,1.0,0.0,0.0] + [0.1,0.1,0.0,0.1] =  1.1, 1.1, 0.0, 0.1]

x has shape of T x C (n_embd) i.e. num of tokens in the current sequence and embedding length for each sequence

This shape x enters the Head as the input.
```

## Step 5 - Create the projections of q, k, and v for each token
```
k = self.key(x), q = self.query(x), v = self.value(x)
Three separate matrices with n_embd x head_size are created.
Weights for Q, K, and V are pre-initialized for each token in vocab.
Assume there are the initializations

W_Q = [  0.1, 0.0,
         0.0, 0.1,
         0.1, 0.1,
         0.0, 0.0
      ]
Each column represents the Q weights for each token in that head.
Column represents head#.
Similarly,
W_K = [[0.2,0.0],[0.0,0.2],[0.0,0.0],[0.2,0.0]]
W_V = [[1.0,0.0],[0.0,1.0],[0.5,0.5],[0.0,0.0]]

Multiplying x @ W_Q, x @ W_K, and x @ W_V row by row i.e. x's column against W_'s rows. 
The_v for head 0 -> 1.1 * 0.1 +  0.0 * 0.0 +  1.0 * 0.1 +  0.1 * 0.0 = 0.21
The_v for head 1 -> 1.1 * 0.0 +  0.0 * 0.1 +  1.0 * 0.1 +  0.1 * 0.0 = 0.10

Therefore, The_v = [0.21, 0.10], and simliarly, cat_v = [0.01, 0.12] and sat_v = [0.11, 0.11].
Similarly, after the calculationis for K and Q, we end up with:
-------------------------------------------------
Token	    q	             k	            v
-------------------------------------------------
The	   [0.21, 0.10]	   [0.24, 0.00]	  [1.60, 0.50]
cat	   [0.01, 0.12]	   [0.20, 0.22]	  [0.05, 1.15]
sat	   [0.11, 0.11]	   [0.24, 0.22]	  [1.10, 1.10]
```

## Step 6 - wei = q @ k.transpose(-2, -1)
```
From step 5, 
Q = [
       0.21, 0.10,
       0.01, 0.12,
       0.11, 0.11
    ]
and K = [
            0.24, 0.00, 
            0.20, 0.22, 
            0.24, 0.22
        ]
Therefore, 
K_Transpose = [
              0.24, 0.20, 0.24
              0.00, 0.22, 0.22   
             ]
The Dot product (@) = 
  [ 
    0.21 * 0.24 + 0.10 * 0.0,  0.2  * 0.20 + 0.10 * 0.22, 0.21 * 0.24 + 0.10 * 0.22
    0.01 * 0.20 + 0.12 * 0.22, 0.01 * 0.20 + 0.12 * 0.22, 0.01 * 0.24 + 0.12 * 0.22,
    0.11 * 0.24 + 0.11 * 0.22, 0.11 * 0.20 + 0.11 * 0.22, 0.11 * 0.24 + 0.11 * 0.22
  ]

gets us the following matrix which maps out the query to key relation for each token
with each other.
          The      cat      sat
The    [0.0504   0.0640   0.0724
cat     0.0024   0.0284   0.0288
sat     0.0264   0.0462   0.0506]

```

## Step 7 Scaling dot product attention
```
scale = 1/sqrt(head_size) = 1/sqrt(2) = 0.7071
Using this on the resulting matrix from step 6 results in this matrix.
Example for r0, c0 val = 0.0504 * 0.7071 = 0.0356

          The      cat      sat
The     0.0356   0.0453   0.0512
cat     0.0017   0.0201   0.0204
sat     0.0187   0.0327   0.0358
```

## Step 8 Casual mask
```
To make sure a position can only see itself and before but not ahead. Here rows = which token is asking, and cols = which token
is it being scored against.
masked_fill(tril==0, -inf) = wei = 
         The      cat      sat
The     0.0356    -inf     -inf
cat     0.0017   0.0201    -inf
sat     0.0187   0.0327   0.0358
```

## Step 9 Applying softmax
```
Use softmax on the last dimension i.e. for each row from Step 8,
normalize across all columns in that row or turn each row into
a probability distribution that sums to 1.
Therefore, softmax(wei, dim = -1) results in: 
        The      cat      sat
The     1.0000   0.0000   0.0000
cat     0.4953   0.5047   0.0000
sat     0.3299   0.3346   0.3356

Here 'The' can only attend to itself, and 'sat' attends to both 'The' and 'cat'.
```
## Step 10 Apply dropout
Dropout is randomness and it's meaningless to show it in a hand-traced calculation. Dropout is also only active 
during the training mode, and not during evaluation.

## Step 11 
```
Described in section 3.2
Turns each token's attention weights into an actual output vector by blending the Value vectors. 
out = wei (From step 9) @ v (from step 5) results in:

       col0    col1
"The"   1.60    0.50
"cat"   0.82    0.83
"sat"   0.91    0.92

Every row of out is weighted average of every token's Value vector. A token whose attention weight is 0 (masked out, low relevance, etc)contrinbutes nothing to the output, regardless of what's in its v vector, and a higher weight contributes to the output.
```


