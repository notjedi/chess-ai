# chess-ai

A chess engine using CNN's + MCTS

I'm trying to build a policy + value head network (an alphazero wannabe)

### Random Thoughts

1) Train DNN on dataset
2) Use DNN to get value and policy outputs
3) Select the best moves and do MCTS on those nodes

- should we encode en passant, promotion and draw by moves?
- will a 5x5 kernel affect the performance of the network? (ig it does cause the chess board is too small(8x8 tensor represenation) 
  and convolving a 5x5 kernel over the image kinda loses the information)

### Acknowledgements

- [FEN](https://www.chess.com/terms/fen-chess)
- [SAN](https://www.chessprogramming.org/Algebraic_Chess_Notation#Standard_Algebraic_Notation_.28SAN.29)
- [python-chess](https://python-chess.readthedocs.io/en/latest/core.html)
