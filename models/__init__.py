from .model import regist_model, get_model
from .ctcn import CTCN

# regist models, sort by alphabet
regist_model("CTCN", CTCN)
