from .reader_utils import regist_reader, get_reader
from .ctcn_reader import CTCNReader

# regist reader, sort by alphabet
regist_reader("CTCN", CTCNReader)
