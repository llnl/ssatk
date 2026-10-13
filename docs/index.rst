Space Situational Awareness Toolkit (SSATK)
===========================================

SSATK is a Python toolkit for space situational awareness (SSA) and
astrodynamics. One package covers orbit and six-degree-of-freedom (6-DoF)
propagation, impulsive and continuous-thrust transfer design, inertial,
Earth-fixed, lunar and satellite frames, the near-Earth space environment,
conjunction screening and orbit determination, standard orbit-data formats, and
visualization from ground tracks to an interactive WebGL satellite viewer.

.. code-block:: bash

   python -m pip install ssatk

.. code-block:: python

   import ssatk

Numerical tests use closed-form results, published values, and separate
implementations such as Astropy, with stated tolerances. Compatibility and
consistency checks have narrower scope; the :doc:`test reference audit
<test_reference_audit>` records the references and remaining limitations. The
:doc:`benchmarking study <benchmarking_ssatk>` positions SSATK against
established astrodynamics and spacecraft-dynamics tools, and the
:doc:`6-DoF design notes <design/6dof_architecture>` explain the spacecraft
body and component model.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   spacecraft_view
   gallery_hosting
   gallery_scene_fidelity
   star_finder
   benchmarking_ssatk
   test_reference_audit
   design/6dof_architecture
   api
