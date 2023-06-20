"""This module at the top level of codebase defines the global settings.

In the code base, for all projects, no absolute path should be used because the
absolute paths can be different for any individual using the codebase unless a
docker is used. Instead, please import this global setting module and create any
path based on the global pathes defined here.

For a user's individual environment, please adjust the global setting here
correspondingly.

if you are using containers, then mount the data and code into workspace.
You can directly use workspace instead of using this global path file.

"""

from etils import epath

DATA_PATH = epath.Path('/workspace/data')
CODEBASE_PATH = epath.Path('/workspace/codebase')
