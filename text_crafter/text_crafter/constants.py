
# import ruamel.yaml as yaml

# root = pathlib.Path(__file__).parent
# for key, value in yaml.safe_load((root / 'data_text.yaml').read_text()).items():
#   globals()[key] = value

# CHANGES MADE TO THE ORIGINAL CODE:
import pathlib
from ruamel.yaml import YAML
root = pathlib.Path(__file__).parent
yaml = YAML(typ='safe', pure=True)
for key, value in yaml.load((root / 'data_text.yaml').read_text()).items():
  globals()[key] = value
