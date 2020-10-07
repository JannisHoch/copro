---
title: 'CoPro: a data-driven model for conflict risk projections'
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
date: 30 September 2020
bibliography: bibliography.bib
---

# Summary

Climate change and environmental degradation are increasingly recognised as factors that can contribute to conflict risk under specific conditions.
In light of predicted shifts in climate patterns and the resulting battle for increasingly scarce resources, it is widely acknowledged that there is a real risk of increased armed conflict. To efficiently plan and implement adaptation and mitigation measures, it is key to first obtain an understanding of the the impact of individual climate, environmental, and societal drivers on conflict risk. And second, conflict risk needs to be projected to a given point in the future to be able to prepare accordingly. With CoPro, a set of functions and workflow is made openly accessible to perform these steps, yielding maps of relative conflict risk. The model caters for a variety of settings and input data, thereby capturing the multitude of facets of the climate-environment-conflict nexus.

# Statement of need 

There is increasing consensus that climate change can exacerbate the risk of (armed) conflict `[@koubi2019climate: @mach2019climate]`. Nevertheless, making (operational) forecasts on the short-term is still challenging due to several reasons `[@cederman2017predicting]`, for example [ADD!]. Building upon recent, similar approaches to use data-driven models `[@colaresi2017robot]` and statistical approaches `[@witmer2017subnational: @hegre2016forecasting]` for conflict risk projections, CoPro is a first, fully open, and extensible tool to combine the inter-disciplinary expertise required to make projections of conflict risk conflict risk associated with climatic and environmental drivers. Such a tool is needed not only to integrate the different disciplines, but also to extend the modelling approach with new insights and data - after all, the established links between climate and societal factors with conflict are still weak, yet positive `[@koubi2019climate: @mach2019climate]`. In addition to scholarly explorations of the inter-dependencies and importance of various conflict drivers, model output such as conflict risk maps can be an invaluable input to inform the decision-making process in affected regions.

Since conflicts are of all times and not limited to specific regions or countries, CoPro is designed with user-flexibility in mind. Therefore, the number and variables provided to the model is not specified, allowing for bespoke model designs. Depending on the modelling exercise and data used, several machine-learning models and pre-processing algorithms are available in CoPro. Catering for different model designs is of added value because of the non-linear and sometimes irrational nature of conflict onset. On top of that, the analyses can be run at any spatial scale, allowing for a better identification of sub-national drivers of conflict risk. After all, conflict onset and conflicts are often limited to specific areas where driving factors conincide, rather than that they encompass entire states. 

Since the replicability of scientific results is important when developing forecast and projection models [@hegre2017introduction], a key concept in mind when establishing CoPro was to be able to produce reproducible output using transparent models. Hence, by making model code openly available and by including dedicated features in the model, we hope to make an important contribution to the existing body of tools developed to project conflict.

These functionalities altogether make CoPro suited for both quick and in-depth analyses of the relative importance of climate, environmental, and societal drivers as well as for assessments how conflict risk can change both in time and space.

# Acknowledgements
This research was supported by a Pathways to Sustainability Acceleration Grant from the Utrecht University.
We kindly acknoweldge the valuable contributions from all partners at PBL, PRIO (Peace Research Institute Oslo), Uppsala University, and Utrecht University.
Last, the valuable comments made by the reviewer are acknowledged.

# References
