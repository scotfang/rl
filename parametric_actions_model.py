from gym.spaces import Discrete, Dict, Box, MultiDiscrete
from ray.rllib.utils.spaces.repeated import Repeated

from ray.rllib.agents.dqn.dqn_torch_model import \
    DQNTorchModel
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.preprocessors import DictFlatteningPreprocessor
from ray.rllib.utils.framework import try_import_torch

# from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX TODO not sure why this line fails
FLOAT_MIN = -3.4e38
FLOAT_MAX = 3.4e38

torch, nn = try_import_torch()

class TorchParametricActionsModel(DQNTorchModel):
    """PyTorch version of above ParametricActionsModel."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 action_embed_size=2,  # Dimensionality of sentence embeddings  TODO don't make this hard-coded
                 **kw):
        DQNTorchModel.__init__(self, obs_space, action_space, num_outputs,
                               model_config, name, **kw)

        self.true_obs_preprocessor = DictFlatteningPreprocessor(obs_space.original_space["true_obs"])
        self.action_embed_model = TorchFC(
            Box(-10, 10, self.true_obs_preprocessor.shape), action_space, action_embed_size,
            model_config, name + "_action_embed")

    @staticmethod
    def make_obs_space(embed_dim=768, max_steps=None, max_utterances=5, max_command_length=5, max_variables=10, max_actions=10, **kwargs):
        true_obs = {
            'dialog_history': Repeated(
                Dict({
                    'sender': Discrete(3),
                    'utterance': Box(-10, 10, shape=(embed_dim,))
                }), max_len=max_utterances
            ),
            'partial_command': Repeated(
                Box(-10, 10, shape=(embed_dim,)), max_len=max_command_length
            ),
            'variables': Repeated(
                Box(-10, 10, shape=(embed_dim,)), max_len=max_variables
            ),
        }
        if max_steps:
            true_obs['steps'] = Discrete(max_steps)

        # return Dict(true_obs) For calculating true_obs_shsape

        return Dict({
            "true_obs": Dict(true_obs),
            '_action_mask': MultiDiscrete([2 for _ in range(max_actions)]),
            '_action_embeds': Box(-10, 10, shape=(max_actions, embed_dim)),
        })

    @staticmethod
    def make_action_space(max_actions=10, **kwargs):
        return Discrete(max_actions)

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        avail_actions = input_dict['obs']["_action_embeds"]
        action_mask = input_dict["obs"]["_action_mask"]
    
        import pdb; pdb.set_trace()
        true_obs = input_dict["obs"]["true_obs"]
        true_obs = self.true_obs_preprocessor.transform(true_obs)
        # Compute the predicted action embedding
        action_embed, _ = self.action_embed_model({
            "obs": true_obs
        })

        # Expand the model output to [BATCH, 1, EMBED_SIZE]. Note that the
        # avail actions tensor is of shape [BATCH, MAX_ACTIONS, EMBED_SIZE].
        intent_vector = torch.unsqueeze(action_embed, 1)

        # Batch dot product => shape of logits is [BATCH, MAX_ACTIONS].
        action_logits = torch.sum(avail_actions * intent_vector, dim=2)

        # Mask out invalid actions (use -inf to tag invalid).
        # These are then recognized by the EpsilonGreedy exploration component
        # as invalid actions that are not to be chosen.
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()
