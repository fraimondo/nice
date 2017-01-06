.. -*- mode: rst -*-

`NICE Tools`
=======================================================

Get the latest code
^^^^^^^^^^^^^^^^^^^

To get the latest code using git, simply type::

    git clone git://github.com/nice-tools/nice.git

If you don't have git installed, you can download a zip or tarball
of the latest code: https://github.com/nice-tools/nice/archives/master

Install nice
^^^^^^^^^^^^^^^^^^

As any Python packages, to install NICE, go in the nice source
code directory and do::

    python setup.py install

or if you don't have admin access to your python setup (permission denied
when install) use::

    python setup.py install --user

You can also install the latest latest development version with pip::

    pip install -e git+https://github.com/nice-tools/nice#egg=nice-dev --user

Dependencies
^^^^^^^^^^^^

The required dependencies to build the software are:
* python >= 2.7 | python >= 3.4
* scipy == 0.18.1
* numpy==1.11.1

And principally, mne-python >= 0.13:
http://mne-tools.github.io/stable/index.html


Some functions require pandas >= 0.7.3.

To run the tests you will also need nose >= 0.10.

Optimizations
^^^^^^^^^^^^^

Aditionally, we ship optimized versions of some algorithms. To build, just
go to the nice soure code directory and do::

    make

Running the test suite
^^^^^^^^^^^^^^^^^^^^^^

To run the test suite, you need nosetests and the coverage modules.
Run the test suite using::

    nosetests

from the root of the project.

Cite
^^^^

If you use this code in your project, please cite::

    *Denis Engemann, *Federico Raimondo, Jean-Remi King, Mainak Jas, Alexandre Gramfort, Stanislas Dehaene, Lionel Naccache, Jacobo Sitt
    "Automated Measurement and Prediction of Consciousness in Vegetative and Minimally Conscious Patients"
    in ICML Workshop on Statistics, Machine Learning and Neuroscience (Stamlins 2015)

Licensing
^^^^^^^^^

NICE is **BSD-licenced** (3 clause):

    This software is OSI Certified Open Source Software.
    OSI Certified is a certification mark of the Open Source Initiative.

    Copyright (c) 2011, authors of MNE-Python
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the names of NICE authors nor the names of any
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    **This software is provided by the copyright holders and contributors
    "as is" and any express or implied warranties, including, but not
    limited to, the implied warranties of merchantability and fitness for
    a particular purpose are disclaimed. In no event shall the copyright
    owner or contributors be liable for any direct, indirect, incidental,
    special, exemplary, or consequential damages (including, but not
    limited to, procurement of substitute goods or services; loss of use,
    data, or profits; or business interruption) however caused and on any
    theory of liability, whether in contract, strict liability, or tort
    (including negligence or otherwise) arising in any way out of the use
    of this software, even if advised of the possibility of such
    damage.**
