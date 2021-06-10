# chess-ai

A chess engine using CNN's + MCTS

I'm trying to build a policy + value head network (an alphazero wannabe)

### what i'm trying to do here:

    general idea

    - train the NN on FICS dataset - on progress
    - while playing select the next move using MCTS
    - instead of doing rollouts on every leaf node pass the state to the NN
    - get value and policy outputs from NN and backpropagate the values through the tree
    - do this idk maybe 500 times and choose the next promising move

    CNN
    
    - a NN with 10 hidden layers (a little deep inspired by AlphaZero and ImageNet)
    - a policy head - probablity of moves
    - a value head  - the end result

    MCTS
    
    - Selection - using UCT
    - Expansion - all the legal moves
    - Playouts - use the NN here
    - Backpropagate - update the different variables for all nodes(that we traversed thro)

    Random Thoughts

    - should we encode en passant, promotion and draw by moves?
    - will a 5x5 kernel affect the performance of the network? (ig it does cause the chess board is too small(8x8 tensor represenation) 
      and we will kinda lose some information while convolving a 5x5 kernel over the image)

### Resources

- [FEN](https://www.chess.com/terms/fen-chess)
- [SAN](https://www.chessprogramming.org/Algebraic_Chess_Notation#Standard_Algebraic_Notation_.28SAN.29)
- [python-chess](https://python-chess.readthedocs.io/en/latest/core.html)
- [MCTS](https://int8.io/monte-carlo-tree-search-beginners-guide/) guide
- [A2C](https://www.datahubbs.com/two-headed-a2c-network-in-pytorch/) network basic guide
- a thread on [BatchNorm](https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/)
