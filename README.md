# ActorCriticControl
Code for the paper [Solving Time-Continuous Stochastic Optimal Control Problems: Algorithm Design and Convergence Analysis of Actor-Critic Flow](https://arxiv.org/pdf/2402.17208).

Examples include the linear quadratic (LQ) problem and the Aiyagari’s growth model.
Run the following command
```
python main.py --config ./configs/LQ1d.json
```
to test the LQ problem in 1 dimension. Other configs include LQ2d.json, LQ5d.json, LQ10d.json, Aiyagari2d.json.