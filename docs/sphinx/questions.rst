FAQ
###


Configuration
=============

**How did you configure DeepLabCut to give multiple labels for a given keypoint in a single frame?**

This is a `contribution <https://github.com/AlexEMG/DeepLabCut/pull/321>`_ that we made to DeepLabCut a while back.
You need to add a ``num_outputs`` parameter to your DeepLabCut ``config.yaml`` file.
For instance, to get 20 labels for a given keypoint per frame, add ``num_outputs = 20``.


**How to use the autoencoder that is mentioned in the Anipose preprint?**

You can train an autoencoder easily by running  ``anipose train-autoencoder`` from your Anipose folder.
This will create an ``autencoder.pickle`` file in your main project folder.
To use this autoencoder, you need to add the following to your ``config.toml``:

.. code:: toml
   
   [filter]
   type = "autoencoder"
   autoencoder_path = "autoencoder.pickle"


Camera setup
============

**Does this work with 1 camera?**

No. We cannot triangulate the detected points with only 1 camera.

**Does this work with 2 cameras?**

Yes, the mouse dataset in the preprint was triangulated with only 2 cameras.

**What is the maximum number of cameras?**

It is unclear. We have tested it with up to 6 cameras, but in principle it should work with any number.

There is `a report <https://github.com/lambdaloop/anipose/issues/21>`_ that calibration works with up to 10 cameras but fails with 11+ cameras. 


**Are there benefits to using more cameras?**

Yes. Although we have not fully characterized this, our scientific peers have!

There are reports that increasing the number of cameras decreases the error by about 10-30% with each camera, with diminishing returns with more cameras. We've seen this on a `human dataset (Figure 2) <https://saic-violet.github.io/learnable-triangulation/>`_ and `fly dataset (Author response image 2) <https://elifesciences.org/articles/48571#SA2>`_

**What kind of resolution do I need for my cameras?**

If you can track it with DeepLabCut, it should work with Anipose!

We found that the error in joint position tracking was <12 pixels in 75% of the frames, and <18 pixels in 90% of the frames consistently across various scales. Thus, if in your dataset 12 pixel ~= 1mm then you would expect errors below 1mm in 75% of your frames. This suggests zooming in on your animal as much as possible and recording at higher resolutions, so that your pixel size is smaller, thus reducing your tracking error.

**Do you need to synchronize the cameras when acquiring?**

Yes, the frames from all the cameras must be acquired simultaneously. There are many ways to achieve this. One way is to use a hardware trigger (as is possible with Basler cameras), which allows for very precise synchronization. Another way is to use a software trigger, in which the computer sends a digital signal to each camera to retrieve an image (as is necessary when using cheap webcams, for instance). Another way is to record video and audio from multiple cameras, and then sync them by synchronizing the audio channel. We have gotten good results from each of these.

**If you calibrate every time, could you just move the camera around, say on a pair or three stands?**

Yes. In some tests we've placed cameras on tripods and recorded multiple videos with different placement on different days. We were able to track the subject in each case.


**Does this work with a mirrored setup, where one camera records two views through a mirror?**

It should be possible to do it with a mirror if you crop the video and flip it and
just treat the 2 cropped parts as separate cameras.
We haven't tried this so we don't have further recommendations yet.
If you have made such a setup work with Anipose, let us know and we can update this question!

**How did you record the hand demo for the tutorial?**

We recorded the hand using some cheap cameras (`EKEN H9R <https://www.amazon.com/EKEN-Waterproof-1080p60-Mountings-Batteries/dp/B01LAIBF2M>`_).
Each camera comes with a remote and they all share the same protocol so the remote triggers all cameras at the same time. We recorded at 720p at 120Hz. We synced the videos more precisely by aligning the videos based on the sound after recording.
