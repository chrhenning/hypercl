# TODOs

* **Migrate all main networks into this folder and deprecate the old once that still reside in subpackages all over the repository**
* The [hnet regularizer](../utils/hnet_regularizer) can't deal with online computation of targets correctly if `hyper_shapes_distilled` is non-empty for the main network. Note, the distillation target for the most recent task should always be obtained via method `distillation_targets` after training on that task.
* Allow the [MLP](mlp.py) network (and possibly other networks implementing gain modulation) to checkpoint gain mod weights.
* In networks that use batchnorm (especially the resnet), it might not make a lot of sense to include bias terms in the conv layers, since batchnorm already includes a bias.
* Add our LeNet implementation to the interface, such that we have a conv net for MNIST and a very simple conv net for CIFAR.
* We need an easy way to map from indices within `param_shapes` to the meaning of the corresponding tensor (i.e., bias, weight matrix, batchnorm weights, context-mod-weight, ...). This is important when implementing an main-net-architecture dependend initialization for the hypernetwork.
* Additionally, it would be nice if there is an easy way to map shapes to their corresponding objects within the `weights`, `batchnorm_layers` or `context_mod_layers` attributes. E.g., if one looks at a bias vector and needs to navigate to the corresponding weight tensor, then he can just look at the corresponding index within the `layer_weight_tensors` attribute.
