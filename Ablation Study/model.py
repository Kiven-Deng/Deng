# New: exact A0 path plus independently trained A1-A5, P, M and D variants.
# Shared parameter names match the unchanged local v3 snapshot.
from dataclasses import replace
import torch
from reference_v3.model import RecoveryTransformer, ModelConfig

VARIANTS = {
    'A0': {'description': 'Full v3 reference'},
    'A1': {'description': 'Remove rank_skip only', 'skip': False},
    'A2': {'description': 'Remove summary_map and rank_skip', 'summary': False, 'skip': False},
    'A3': {'description': 'Mean rank repeated across observations', 'mean_only': True},
    'A4': {'description': 'Remove learned observation branch', 'observation': False},
    'A5': {'description': 'Remove encoder attention residual only', 'encoder_attention': False},
    'P10': {'description': 'Rank-log RoPE only', 'rarope': True},
    'P01': {'description': 'ANLiBi-style bias only', 'anlibi': True},
    'P11': {'description': 'Both position mechanisms', 'rarope': True, 'anlibi': True},
    'M10': {'description': 'Train with m=10; validate on full grid', 'train_m': [10]},
    'M2': {'description': 'Train with m=2; validate on full grid', 'train_m': [2]},
    'Duniform': {'description': 'Matched-strength uniform training noise', 'data': 'uniform'},
    'Dstage': {'description': 'Stagewise training noise', 'data': 'stage'},
}
GROUPS = {'core': list(VARIANTS)[:6], 'position': ['A0','P10','P01','P11'],
          'm': ['A0','M10','M2'], 'data': ['A0','Duniform','Dstage'], 'all': list(VARIANTS)}

class AblationTransformer(RecoveryTransformer):
    def __init__(self, config=None, variant='A0'):
        if variant not in VARIANTS: raise ValueError(variant)
        flags = VARIANTS[variant]
        config = replace(config or ModelConfig(), rarope=flags.get('rarope', False),
                         anlibi=flags.get('anlibi', False))
        super().__init__(config)
        self.variant, self.flags = variant, flags
        # Zero the input of the existing Linear to preserve parameter names.
        # The hook only implements the skip gate; it carries no parameters.
        self.rank_skip.register_forward_pre_hook(self._skip_input)

    def _skip_input(self, module, args):
        return args if self.flags.get('skip', True) else (torch.zeros_like(args[0]),)

    def encode(self, positions):
        if self.flags.get('mean_only', False):
            positions = positions.mean(-1, keepdim=True).expand_as(positions)
        observation, summary = self.features(positions)
        if self.flags.get('observation', True): hidden = self.observation_map(observation).mean(2)
        else: hidden = positions.new_zeros((*positions.shape[:2], self.config.hidden))
        if self.flags.get('summary', True): hidden = hidden + self.summary_map(summary)
        ranks = positions.mean(-1)
        for encoder in self.encoders:
            if self.flags.get('encoder_attention', True): hidden = encoder(hidden, ranks)
            else: hidden = hidden + encoder.ff(encoder.norm2(hidden))
        return hidden, ranks, summary

    def active_parameter_count(self):
        excluded = []
        if not self.flags.get('skip', True): excluded.append('rank_skip.')
        if not self.flags.get('summary', True): excluded.append('summary_map.')
        if not self.flags.get('observation', True): excluded.append('observation_map.')
        def active(name):
            if any(name.startswith(p) for p in excluded): return False
            if not self.flags.get('encoder_attention', True) and name.startswith('encoders.'):
                if '.attention.' in name or '.norm1.' in name: return False
            return True
        return sum(p.numel() for name,p in self.named_parameters() if active(name))

def initialize(config, variant, shared_state):
    model = AblationTransformer(config, variant)
    own = model.state_dict()
    if not set(own) <= set(shared_state): raise ValueError('Initialization is incomplete')
    model.load_state_dict({k: shared_state[k].clone() for k in own}, strict=True)
    return model

def shared_initialization(config, seed):
    # Build the parameter superset once. All common tensors are copied verbatim.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = RecoveryTransformer(replace(config, rarope=True, anlibi=True))
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

