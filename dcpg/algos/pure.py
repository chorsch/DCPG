from collections import deque
import copy

import numpy as np

import torch
from torch.distributions import kl_divergence
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler

from dcpg.algos import DCPG

class InverseDynamicsNet(nn.Module):
    def __init__(self, num_actions, emb_size=256, hidden_dim=256):
        super(InverseDynamicsNet, self).__init__()
        self.num_actions = num_actions 
        
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(2 * emb_size, hidden_dim), 
            nn.ReLU(),
        )
        self.id_out = nn.Linear(hidden_dim, self.num_actions)

        
    def forward(self, state_embedding, next_state_embedding):
        inputs = torch.cat((state_embedding, next_state_embedding), dim=-1)
        action_logits = self.id_out(self.inverse_dynamics(inputs))
        return action_logits

class PureExploration(DCPG):
    """
    Delayed Critic Policy Gradient (DCPG)
    """

    def __init__(
        self,
        actor_critic,
        feature_encoder,
        num_actions,
        # PPO params
        clip_param,
        ppo_epoch,
        num_mini_batch,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        # Aux params
        buffer_size=None,
        aux_epoch=None,
        aux_freq=None,
        aux_num_mini_batch=None,
        policy_dist_coef=None,
        value_dist_coef=None,
        # Misc
        device=None,
        feature_encoder_embed_dim=256,
        **kwargs
    ):
        super().__init__(
            actor_critic=actor_critic,
            # PPO params
            clip_param=clip_param,
            ppo_epoch=ppo_epoch,
            num_mini_batch=num_mini_batch,
            entropy_coef=entropy_coef,
            lr=lr,
            eps=eps,
            max_grad_norm=max_grad_norm,
            # Aux params
            buffer_size=buffer_size,
            aux_epoch=aux_epoch,
            aux_freq=aux_freq,
            aux_num_mini_batch=aux_num_mini_batch,
            policy_dist_coef=policy_dist_coef,
            value_dist_coef=value_dist_coef,
            # Misc
            device=device,
            **kwargs
        )
        self.feature_encoder = feature_encoder

        self.inverse_dynamics_model = InverseDynamicsNet(num_actions, emb_size=feature_encoder_embed_dim)
        self.inverse_dynamics_model.to(device)

        idm_params = list(self.feature_encoder.parameters()) + list(self.inverse_dynamics_model.parameters())
        self.idm_optimizer = optim.Adam(idm_params, lr=lr, eps=eps)

    def update(self, on_policy_rollouts, full_rollouts):
        # Add obs and returns to buffer
        seg = {key: on_policy_rollouts[key].cpu() for key in ["obs", "returns"]}
        self.buffer.insert(seg, on_policy_rollouts.step)

        # PPO phase
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        value_dist_epoch = 0

        # Clone actor-critic
        old_actor_critic = copy.deepcopy(self.actor_critic)

        for _ in range(self.ppo_epoch):
            data_generator = on_policy_rollouts.feed_forward_generator(
                num_mini_batch=self.num_mini_batch
            )

            for sample in data_generator:
                # Sample batch
                (
                    obs_batch,
                    _,
                    actions_batch,
                    old_action_log_probs_batch,
                    _,
                    _,
                    _,
                    _,
                    adv_targs,
                    _,
                ) = sample

                # Feed batch to actor-critic
                actor_outputs, critic_outputs = self.actor_critic(obs_batch)
                dists = actor_outputs["dist"]
                values = critic_outputs["value"]

                # Feed batch to old actor-critic
                with torch.no_grad():
                    old_critic_outputs = old_actor_critic.forward_critic(obs_batch)
                old_values = old_critic_outputs["value"]

                # Compute action loss
                action_log_probs = dists.log_probs(actions_batch)
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                ratio_clipped = torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surr1 = ratio * adv_targs
                surr2 = ratio_clipped * adv_targs
                action_loss = -torch.min(surr1, surr2).mean()
                dist_entropy = dists.entropy().mean()

                # Compute value dist
                value_dist = 0.5 * (values - old_values).pow(2).mean()

                # Update parameters  
                self.policy_optimizer.zero_grad()
                loss = (
                    action_loss
                    - dist_entropy * self.entropy_coef
                    + value_dist * self.value_dist_coef
                )
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.policy_optimizer.step()

                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                value_dist_epoch += value_dist.item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        value_dist_epoch /= num_updates

        self.num_policy_updates += 1

        # Aux phase
        if self.num_policy_updates % self.aux_freq == 0:
            value_loss_epoch = 0
            policy_dist_epoch = 0

            # Clone actor-critic
            old_actor_critic = copy.deepcopy(self.actor_critic)

            for _ in range(self.aux_epoch):
                data_generator = self.buffer.feed_forward_generator(
                    num_mini_batch=self.aux_num_mini_batch
                )

                for sample in data_generator:
                    # Sample batch
                    obs_batch, returns_batch = sample

                    # Feed batch to actor-critic
                    actor_outputs, critic_outputs = self.actor_critic(obs_batch)
                    dists = actor_outputs["dist"]
                    values = critic_outputs["value"]

                    # Feed batch to old actor-critic
                    with torch.no_grad():
                        old_actor_outputs = old_actor_critic.forward_actor(obs_batch)
                    old_dists = old_actor_outputs["dist"]

                    # Compute value loss
                    value_loss = 0.5 * (values - returns_batch).pow(2).mean()

                    # Compute policy dist
                    policy_dist = kl_divergence(old_dists, dists).mean()

                    # Update parameters
                    self.aux_optimizer.zero_grad()
                    loss = value_loss + policy_dist * self.policy_dist_coef
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), self.max_grad_norm
                    )
                    self.aux_optimizer.step()

                    value_loss_epoch += value_loss.item()
                    policy_dist_epoch += policy_dist.item()

            num_updates = self.aux_epoch * self.aux_num_mini_batch * len(self.buffer)
            value_loss_epoch /= num_updates
            policy_dist_epoch /= num_updates

            self.prev_value_loss_epoch = value_loss_epoch
            self.prev_policy_dist_epoch = policy_dist_epoch
        else:
            value_loss_epoch = self.prev_value_loss_epoch
            policy_dist_epoch = self.prev_policy_dist_epoch

        # Inverse dynamics training
        idm_loss_epoch = 0
        for _ in range(self.ppo_epoch):
            self.idm_optimizer.zero_grad()
            inputs = full_rollouts["obs"]
            actions = full_rollouts["actions"]
            dones_masks = full_rollouts["masks"]
            actions = actions.view(-1)
            dones_masks = dones_masks[:-1].view(-1)
            
            phi = self.feature_encoder(inputs.view(-1, *inputs.size()[2:]))
            phi = phi.view(inputs.shape[0], inputs.shape[1], -1)
            phi_step_t = phi[:-1]
            phi_step_tp1 = phi[1:]
            phi_step_t = torch.flatten(phi_step_t, 0, 1)
            phi_step_tp1 = torch.flatten(phi_step_tp1, 0, 1)
            pred_action_logits = self.inverse_dynamics_model(phi_step_t, phi_step_tp1)
            pred_action_logits = torch.log_softmax(pred_action_logits, dim=1)
            inverse_dynamics_loss = nn.functional.nll_loss(pred_action_logits, actions, reduction='none')
            inverse_dynamics_loss *= dones_masks
            inverse_dynamics_loss = torch.sum(inverse_dynamics_loss) / dones_masks.sum()
            
            inverse_dynamics_loss.backward()

            if self.max_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(self.feature_encoder.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.inverse_dynamics_model.parameters(), self.max_grad_norm)

            
            self.idm_optimizer.step()
            idm_loss_epoch += inverse_dynamics_loss.item()
        idm_loss_epoch /= self.ppo_epoch

        train_statistics = {
            "action_loss": action_loss_epoch,
            "dist_entropy": dist_entropy_epoch,
            "value_loss": value_loss_epoch,
            "policy_dist": policy_dist_epoch,
            "value_dist": value_dist_epoch,
            "idm_loss": idm_loss_epoch,
        }

        return train_statistics
