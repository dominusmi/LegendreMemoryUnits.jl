# LegendreMemoryUnits.jl

Julia implementation for the [Legendre Memory Units](https://papers.nips.cc/paper/9689-legendre-memory-units-continuous-time-representation-in-recurrent-neural-networks.pdf)

The training performance are currently very low, potentially due to bug in Flux.jl library with RNNs.
Issue open: https://github.com/FluxML/Flux.jl/issues/980 

Python implementation provided by authors: https://github.com/abr/neurips2019

## Original paper citation
```
@inproceedings{voelker2019lmu,
  title={Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks},
  author={Aaron R. Voelker and Ivana Kaji\'c and Chris Eliasmith},
  booktitle={Advances in Neural Information Processing Systems},
  pages={15544--15553},
  year={2019}
}
```
