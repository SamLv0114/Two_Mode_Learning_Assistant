---
id: graph_neural_networks
name: Graph neural networks
triggers: graph neural network, gnn, graph convolutional, graph convolution, gcn, graph attention network, message passing neural network, message-passing gnn, node classification
---
## Often left open
- How the adjacency matrix is normalised: symmetric D̃^−1/2·Ã·D̃^−1/2 as in GCN, or a row-normalised mean.
- Whether self-loops are added, the number of layers, and the aggregator (sum, mean, max or attention).

## Common mistakes
- Self-loops left out, so a node ignores its own features. GCN uses Ã = A + I.
- Degrees computed from A instead of from Ã after the self-loops are added.
- Graph-attention scores normalised over all nodes instead of over each node's neighbours.
- Edge direction ignored or reversed when the paper's graph is directed.

## Self-checks
- Permuting the nodes permutes the outputs in the same way.
- For an undirected graph the symmetric-normalised adjacency matrix is symmetric.
- An isolated node's output depends only on its own features.
