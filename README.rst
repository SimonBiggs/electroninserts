========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires|
        | |codecov|
    * - package
      - |version| |downloads| |wheel| |supported-versions| |supported-implementations|

.. |docs| image:: https://readthedocs.org/projects/electroninserts/badge/?style=flat
    :target: https://readthedocs.org/projects/electroninserts
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/SimonBiggs/electroninserts.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/SimonBiggs/electroninserts

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/SimonBiggs/electroninserts?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/SimonBiggs/electroninserts

.. |requires| image:: https://requires.io/github/SimonBiggs/electroninserts/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/SimonBiggs/electroninserts/requirements/?branch=master

.. |codecov| image:: https://codecov.io/github/SimonBiggs/electroninserts/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/SimonBiggs/electroninserts

.. |version| image:: https://img.shields.io/pypi/v/electroninserts.svg?style=flat
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/electroninserts

.. |downloads| image:: https://img.shields.io/pypi/dm/electroninserts.svg?style=flat
    :alt: PyPI Package monthly downloads
    :target: https://pypi.python.org/pypi/electroninserts

.. |wheel| image:: https://img.shields.io/pypi/wheel/electroninserts.svg?style=flat
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/electroninserts

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/electroninserts.svg?style=flat
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/electroninserts

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/electroninserts.svg?style=flat
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/electroninserts


.. end-badges

Deprecation notice
=======

This application is deprecated and no longer in use by the author. All future development of the electron model will be undergone within the pymedphys package:

> https://github.com/CCA-Physics/pymedphys/blob/master/pymedphys/level1/electronfactors.py

Licence
=======
A simplified more transparent package for the spline modelling of electron insert factors

Copyright (C) 2016, Simon Grant Biggs

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.



Installation
============

::

    pip install electroninserts

Documentation
=============

https://electroninserts.readthedocs.org/

Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
