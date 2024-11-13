# Introduction
Transformers was first introduced by the paper "Attention is all you need" by Vaswani et. al. The most important piece of the transformer architecture
is the concept of attention (make this bold). Attention, enables the transformer architecture to manage and understand long-term dependency. Considering
tensors, the attention tensor consists of continuous values that showcase the weights that the block/model is focusing on. The indices with high values, indicate
more attention is being paid to that token.

# Model Architecture
Looking at the below figure (Need to have a transformer architecture figure) which is taken from the original paper, one can see the architecture can be divided into
an encoder and a decoder block.

1) Encoder:
    - Every layer has two sub-layers:
        - Sub-layer one is the multi-head attention block
        - Sub-layer two is the fully connected feed-forward network.
        - There are residual connections and layer normalizations. Because of the residual connection LayerNorm(x+sublayer(x)), it is important to maintain and match   dimensions.
        - Nx simply represents stacking the same layer 'N' number of times.

2) Decoder:
    - Along with two sub-layers that come from the encoder, there is an additional sub-layer
        - Additional sub-layer is the multi-head attention over the output of the encoder
        - Residual connections and layer normalizations just like the encoder architectures
        - There is a modification of the self-attention sub-layer by shifting right
            - Done to prevent positions from attending to subsequent positions
            - Predictions for output 'i' (make this italic) can depend only on the known outputs at positions less than 'i'

## Individual architecture blocks
1) Attention:
    - An attention function, as described by the paper, is mapping a query and a set of key-value pairs to an output.
    - Key, Query and value are all vectors
    - The output is a weighted sum of the values, where the weight assigned to each value is computed by a compatibilty function
        - This compatibility function is between a query with the corresponding key
    
    - The authors use the scaled-dot product attention
        Attention (Q,K,V) = (Have actual formula here)
    
    - One mention in the paper as to why they use root(d_k) is w.r.t scaling
        - For large values of d_k, the softmax function will result in extremely small gradients and therefore the scaling factor will help


# TODO
1) Change the architectural blocks to its specific folders
2) Add unit tests to check the functionality of each of the block
3) Update documentation of the Readme.md file and other files.