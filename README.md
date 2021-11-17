# chess-ai

A chess engine using CNN's + MCTS

I'm trying to build a policy + value head network (an alphazero wannabe - the self play)

### what i'm trying to do here:

    general idea

    - train the NN on FICS dataset
    - while playing select the next move using MCTS
    - instead of doing rollouts on every leaf node pass the state to the NN
    - get value and policy outputs from NN and backpropagate the node stats up the tree

    CNN
    
    - a NN with 10 hidden layers (1 conv + 10 resnets)
    - a policy head - probablity of moves
    - a value head  - the end result

    MCTS
    
    - Selection - using UCT
    - Expansion - all the legal moves
    - Playouts - use the NN here
    - Backpropagate - update the different variables for all nodes(that we traversed thro)

    TODO

    - There is definitely something wrong with my MCTS implementation (raw outputs from 
      NN perform better), fix it
    - Train the model on more data? I was able to go upto 0.4 on a sample subset of data
      the current best model is about 1.7
    - try out 5x5 conv in the first layer

    OBSERVATIONS

    - Weight decay (i used 0.1) destorys the model
    - Moving from a 64-solution (each cell) space to 1968-solution (each possible move) space 
      (taking into account the totality of moves) helps the model tremendously
    - Using lr scheduler doesn't help much (probably plateau scheduler would help, haven't tried it though)
    - use float16, get memory discount
    - max out batch size, save training time


### Resources

- [FEN](https://www.chess.com/terms/fen-chess)
- [SAN](https://www.chessprogramming.org/Algebraic_Chess_Notation#Standard_Algebraic_Notation_.28SAN.29)
- [python-chess](https://python-chess.readthedocs.io/en/latest/core.html)
- [MCTS](https://int8.io/monte-carlo-tree-search-beginners-guide/) guide
- a thread on [BatchNorm](https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/)
