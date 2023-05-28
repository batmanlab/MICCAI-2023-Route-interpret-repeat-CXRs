Thank you for engaging in discussion with us; we appreciate it.

In response, we would like to point out the following:

* Our method is closer to the interpretable approaches because it requires training a new model. We are following the
  experiment design in that category [1-6], where the evaluation is against a BlackBox model, and the metric is how much
  the interpretable design hurts the classification performance. We already have that results (see Fig. 2). None
  of [1-6] compare against CAV because CAV is a post hoc explanation method (involving no training).

> "... that the explanations are of better quality than for post-hoc explainability."

* Arguably, the posthoc explanation methods can be divided into feature attribution (eg GradCAM) that generates feature
  importance heatmap and concept-based explanation. The feature attribution approaches are not applicable because they
  don't use concepts. The closer ones are concepts-based post hoc explanations such as CAV. We are not aware of any
  paper that provides a protocol to compare the quality of explanation between an interpretable method (like ours) and
  post hoc explanations. Among the post hoc explanations method, TCAV is used for comparison that requires a pre-trained
  model. This is not applicable to our method because our method requires training a new network. Furthermore, TCAV is
  criticized for not being a good metric to compare posthoc concept-based explanation (see [7], Fig 5).

* One idea to compare between explanation/interpretable methods is to evaluate whether the explanation is *actionable*
  meaning if concepts are correctly identified, one should be able to intervene and *fix/alter* BlackBox's property. Our
  last experiment is designed for that purpose. We showed that one could use our method to fix the shortcut learning
  issue of a BlackBox. None of the posthoc approaches can do this because there is no mechanism to intervene in the
  BlackBox.

* Finally, for the sanity check, we include the result of the random intervention on the concepts. Such intervention
  should degrade the performance of our model. For example, for CUB-200 VIT-derived MoIE, the performance of MoIE drops
  from 91.30 to 60.13% ($\Delta$=31.17%) (see Table 8 in the Appendix A.9). We did the same experiment for another
  concept-based method [1], and the drop is from 74.80% to 53.16% ($\Delta$=21.64%).

## References

[1] Concept Bottleneck Models, Koh et al.

[2] Post-hoc Concept Bottleneck Models, Yuksekgonul et al.

[3] Entropy-based Logic Explanations of Neural Networks, Barbiero et al.

[4] Concept Embedding Models, Zarlenga et al.

[5] Logic Explained Networks Ciravegna et al.

[6] A Framework for Learning Ante-hoc Explainable Models via Concepts, Sarkar et al.

[7] Concept Activation Regions: A Generalized Framework For Concept-Based Explanations, Crabbé et al.