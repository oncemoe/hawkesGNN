from .bitcoin_otc import BitcoinOTC
from .bitcoin_alpha import BitcoinAlpha
from .metr_la import MetrLa
from .uci_message import UCIMessage
from .as733 import AS733
from .sbm import SBM
from .stackoverflow import StackOverflow
from .reddit_title import RedditTitle
from .reddit_body import RedditBody
from .dblp import DBLP
from .tmall import TMALL

__all__ = [
    'BitcoinOTC',
    'BitcoinAlpha',
    'MetrLa',
    'UCIMessage',
    'AS733',
    'SBM',
    'StackOverflow',
    'RedditTitle',
    'RedditBody',

    'DBLP',
    'TMALL'
]

classes = __all__
