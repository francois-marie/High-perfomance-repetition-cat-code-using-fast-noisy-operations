from pathlib import Path

from qsim._version import VERSION

# from qsim.helpers import all_simulations, search_simulation, simulate


qsim_repo_path = Path(__file__).parent.parent.absolute()
data_directory_path = qsim_repo_path / 'data'
