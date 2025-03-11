===========
Basic Usage
===========

``xsdba`` performs a training on a source dataset `hist` to reproduce a target `ref`. "ref" is short for reference, and "hist" for historical: In climate services, it is common to refer to the training data as historical, a period of time where observations are available. The training is then applied to a dataset `sim` to obtain an adjusted dataset. This may simply be the training data `hist` which is not adjusted in the first training part. An example with a basic Quantile Mapping is given below.

.. code-block:: python

    import xsdba

    # Example: Using a Quantile Mapping method
    # Instantiate a TrainAdjust class with the training, with the source `hist` mapped to a target `ref`
    ADJ = xsdba.EmpiricalQuantileMapping.train(ref=ref, hist=hist)
    # Perform adjust for data outside the training period, `sim`
    adj = ADJ.adjust(sim=sim)
..
