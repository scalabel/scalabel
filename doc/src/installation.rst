Installation Tips
====================

Development
-------------------

We transpile or build Javascript code

.. code-block:: bash

    npm install node_modules/.bin/webpack --config webpack.config.js --mode=production


If you are debugging the code, it is helpful to build the javascript code in
development mode, in which you can trace the javascript source code in your
browser debugger. `--watch` tells webpack to monitor the code changes and
recompile automatically.

.. code-block:: bash

    node_modules/.bin/webpack --watch --config webpack.config.js --mode=development


Upgrade Python
-------------------

Our python code requires Python3.7 and above. To install the proper Python
versions, we recommend [pyenv](https://github.com/pyenv/pyenv), especially for
Mac users.

Homebrew on Mac can directly install `pyenv`

.. code-block:: bash

    brew update && brew install pyenv


Otherwise, you can follow the `pyenv` [installation
tutorial](https://github.com/pyenv/pyenv#installation). Next, install Python
3.8.2

.. code-block:: bash

    pyenv install 3.8.2


Set it as global default

.. code-block:: bash

    pyenv global 3.8.2


Adding the new Python to your `PATH`

.. code-block:: bash

    export PATH=$(pyenv root)/shims:$PATH


On Ubuntu, you can also use
`deadsnakes ppa <https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa/+index>`_
to install different versions of Python. Ubuntu 18.04 or later provides the
package of Python 3.8 directly.

.. code-block:: bash

    sudo apt-get update sudo apt-get install -y python3.8 python3.8-dev \
        python3-pip python3-setuptools


Redis security
-------------------

Please make sure secure your redis server following
https://redis.io/topics/security/. By default redis will backup to local file
storage, so ensure you have enough disk space or disable backups inside
redis.conf.

.. .. ### Cognito Integration

.. .. Scalabel could integrate with [AWS Cognito](https://aws.amazon.com/cognito/).
.. You can use Cognito to manage users. Once you have set up Cognito (See official
.. tutorial
.. [here](https://docs.aws.amazon.com/cognito/latest/developerguide/tutorials.html)),
.. go to config file, fill the properties like below.

.. .. ```yaml .. userManagement: true //If set to true, then the following configs
.. are required .. cognito: ..   region: "us-west-2" ..   userPool:
.. "us-west-2_tgxuoXZdf" ..   clientId: "52i44u3c7fapmec4oaqto4lk121" ..
.. userPoolBaseUri: "scalabel.auth.us-west-2.amazoncognito.com" ..   callbackUri:
.. "http://localhost:8686/callback" .. ```

.. .. - region: Region of your cognito service. .. - userPool: Pool ID - You can
.. find it in [General Settings] .. - clientID: App Client ID - You can find it in
.. [General settings] -> [App clients] or [App integration] -> [App client
.. settings] .. - userPoolBaseUri: App Domain - You can find it in [App
.. integration] -> [Domain name] .. - callbackUri: Must exact as what you filled in
.. [App integration] -> [App client settings]

Backward compatibility
-----------------------

We are doing our best to make sure that our system can be stable and the
internal data can be reused in the new updates. At the same time, we are also
iterating on the internal design so that it can be more efficient and versatile.
In some cases, we have to break the backward compatibility for the internal data
storage. When this happens to your project, you can export the labels from the
old project and import them to the new project with the new code. We definitely
hope you can enjoy the new features we constantly add to Scalabel.