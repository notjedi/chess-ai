import torch

from util import create_uci_labels

LABELS = create_uci_labels()
N_LABELS = len(LABELS)
REPORT = 500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
