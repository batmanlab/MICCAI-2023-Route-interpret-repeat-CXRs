Thank you for raising this point in discussion with us; we appreciate it.

In response, we would like to point out the following:

* First Bau et al. aims to find concept-neuron association but we aim to find how the different concepts are composed
  together for the class prediction. Though both of them deals with concepts, the fundamental objective of the two
  methods are different. However, as suggested in [1, 2], we can leverage the concepts captured in BRODEN dataset by Bau
  et al. to design an interpretable model for datasets where concept annotation is scarce. We first project the
  representation of the blackbox $\Phi$ to the concept space spanned by the concept vectors in Broden dataset. Next, the
  interpretable model consumes the projected concept vectors to predict the class labels. So, we create a concept bank
  using the concepts from BRODEN dataset following [1, 2]. We capture a total of 170 visual concepts as mentioned in
  table 5 in [1] like 'hair' or '
  eyebrow'. As CIFAR10 and CIFAR100 datasets does not have an explicit concept annotation, we utilize these concepts to
  design the Mixture of Interpretable experts (MoIE) for image classification. As a blackbox, we use CLIP-ResNet50 as
  referred in [2]. We follow similar hyperparameter settings as in [1, 2]. Refer below for the comparison of Top-1
  Accuracy of the different models.

| Method                             | CIFAR10 | CIFAR100 |
|------------------------------------|---------|----------|
| Blackbox (ClipResNet50)            | 87.6 %  | 68.1 %   |
| MoIE (99 % coverage)               | 85.4 %  | 66.3 %   |
| Interpretable by design (Baseline) | 83.2 %  | 62.8 %   |

## References

[1] Meaningfully Debugging Model Mistakes using Conceptual Counterfactual Explanations, Abid et al.

[2] Post-hoc Concept Bottleneck Models, Yuksekgonul et al.
