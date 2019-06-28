# TODOs for the Toy Example
* FIX embedding arithmetic experiments.
* Try to learn a task only by learning embeddings (given that similar tasks have been learned before)
* Implement incremental learning, such that the number of tasks doesn't have to be necessarily known in advance (mainly concerns the recognition model, where the softmax is currently always computed across all alphas)
* Implement recognition model that uses hypernetwork in the decoder.
* Use KD loss (knowledge distillation) for recognition model
