FAQ
###


How did you configure DeepLabCut to give multiple labels for a given keypoint in a single frame?
------------------------------------------------------------------------------------------------

This is a `contribution <https://github.com/AlexEMG/DeepLabCut/pull/321>`_ that we made to DeepLabCut a while back.

You need to add a ``num_outputs`` parameter to your DeepLabCut config.yaml .
For instance, to get 20 labels for a given keypoint per frame, add ``num_outputs = 20``.


How to use the autoencoder that is mentioned in the Anipose preprint?
-----------------------------------------------------------------------

You can train an autoencoder easily by running  ``anipose train-autoencoder`` from your Anipose folder.

This will create an ``autencoder.pickle`` file in your main project folder.
To use this autoencoder, you need to add the following to your ``config.toml``:

.. code:: toml
   
   [filter]
   type = "autoencoder"
   autoencoder_path = "autoencoder.pickle"
