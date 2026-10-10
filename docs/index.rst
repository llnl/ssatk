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

Every automated test compares a computed value against an independent
reference (a closed-form result, a published value, or a separate
implementation such as Astropy) with a stated tolerance. The
:doc:`benchmarking study <benchmarking_ssatk>` positions SSATK against
established astrodynamics and spacecraft-dynamics tools, and the
:doc:`6-DoF design notes <design/6dof_architecture>` explain the spacecraft
body and component model.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   benchmarking_ssatk
   design/6dof_architecture
   api
