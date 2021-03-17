---
title: 'CoPro: a data-driven modelling framework for conflict risk projections'
tags:
  - Python
  - climate change
  - projections
  - conflict
  - climate security
  - water
  - risk
authors:
  - name: Jannis M. Hoch^[corresponding author]
    orcid: 0000-0003-3570-6436
    affiliation: 1
  - name: Sophie de Bruin
    orcid: 0000-0003-3429-349X
    affiliation: "1, 2"
  - name: Niko Wanders
    orcid: 0000-0002-7102-5454
    affiliation: 1
affiliations:
 - name: Department of Physical Geography, Utrecht University, Utrecht, the Netherlands
   index: 1
 - name: PBL Netherlands Environmental Assessment Agency, the Hague, the Netherlands
   index: 2
date: 18 February 2021
bibliography: bibliography.bib
---

# Summary

Climate change and environmental degradation are increasingly recognized as factors that can contribute to conflict risk under specific conditions.
In light of predicted shifts in climate patterns and the potentially resulting battle for increasingly scarce resources, it is widely acknowledged that there is an actual risk of increased armed conflict. To efficiently plan and implement adaptation and mitigation measures, it is key to first obtain an understanding of conflict drivers and spatial conflict risk distribution. And second, conflict risk needs to be projected to a given point in the future to be able to prepare accordingly. With CoPro, building and running models investigating the interplay between conflict and climate is made easier. By means of a clear workflow, maps of conflict risk for today as well as the future can be produced. Despite the structured workflow, CoPro caters for a variety of settings and input data, thereby capturing the multitude of facets of the climate-environment-conflict nexus.

# Statement of need 

There is increasing consensus that climate change can exacerbate the risk of (armed) conflict [@koubi2019climate; @mach2019climate]. Nevertheless, making (operational) projections of conflict risk is still challenging due to several reasons [@cederman2017predicting]. Building upon recent, similar approaches to use data-driven models [@colaresi2017robot] and statistical approaches [@witmer2017subnational; @hegre2016forecasting], CoPro is a novel, fully open, and extensible Python-model facilitating the set-up, execution, and evaluation of machine-learning models predicting conflict risk. CoPro provides a structured workflow including pre- and post-processing tools, making it accessible to all levels of experience. Such a user-friendly tool is needed not only to integrate the different disciplines, but also to extend the modeling approach with new insights and data - after all, the established links between climate and societal factors with conflict are still weak [@koubi2019climate; @mach2019climate]. In addition to scholarly explorations of the inter-dependencies and the importance of various conflict drivers, model output such as maps of spatially-disaggregated projected conflict risk can be an invaluable input to inform the decision-making process in affected regions.

Since conflicts are of all times and not limited to specific regions or countries, CoPro is designed with user-flexibility in mind. Therefore, the number and variables provided to the model is not specified, allowing for bespoke model designs. Depending on the modeling exercise and data used, several machine-learning models and pre-processing algorithms are available in CoPro. In its current form, the supervised learning techniques support vector classifier, k-neighbors classifier, and random-forest classifier are implemented. Catering for different model designs is of added value because of the non-linear and sometimes irrational - 'law-breaking' [@cederman2017predicting] - nature of conflicts. On top of that, the analyses can be run at any spatial scale, allowing for better identification of sub-national drivers of conflict risk. After all, conflict onset and conflicts are often limited to specific areas where driving factors coincide. 

Since the replicability of scientific results is important when developing forecast and projection models [@hegre2017introduction], CoPro produces reproducible output using transparent models. Hence, by making model code openly available and by including dedicated features in the model, we hope to advance the existing body of tools developed to project conflict risk.

These functionalities altogether make CoPro suited for both 'quick-and-dirty' and in-depth analyses of the relative importances of climate, environmental, and societal drivers as well as for assessments how conflict risk can change both in time and space.

# Acknowledgements
This research was supported by a Pathways to Sustainability Acceleration Grant from the Utrecht University.
We kindly acknowledge the valuable contributions from all partners at PBL, PRIO (Peace Research Institute Oslo), Uppsala University, and Utrecht University.

# References
