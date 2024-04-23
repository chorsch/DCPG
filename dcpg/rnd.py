import torch as th
import numpy as np
from typing import Tuple

from dcpg.models import ResNetEncoder

class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class RandomNetworkDistillation:
    """This class uses Random Network Distillation to estimate the uncertainty/novelty of state-actions."""
    def __init__(self, 
                 action_space, 
                 observation_space, 
                 embed_dim, 
                 policy_kwargs, 
                 device="cpu", 
                 flatten_input=False, 
                 use_resnet=False, 
                 normalize_images=False, 
                 normalize_output=False,
                 norm_epsilon=1e-12,
                 **kwargs):
        self.criterion = th.nn.MSELoss(reduction="none")
        activation = policy_kwargs["activation_fn"]
        hidden_dims = policy_kwargs["net_arch"]
        learning_rate = policy_kwargs["learning_rate"]
        self.device=th.device(device)
        self.n_actions = action_space.n
        self.use_resnet = use_resnet
        self.normalize_images = normalize_images
        self.normalize_output = normalize_output
        self.norm_epsilon = norm_epsilon

        if normalize_output:
            self.rnd_rms = RunningMeanStd(shape=())

        self.target_net = []
        self.predict_net = []

        if self.use_resnet:
            self.target_cnn = ResNetEncoder(observation_space.shape, feature_dim=1024).to(th.device(device))
            self.predict_cnn = ResNetEncoder(observation_space.shape, feature_dim=1024).to(th.device(device))
            with th.no_grad():
                n_flatten = np.prod(self.target_cnn(th.as_tensor(observation_space.sample()[None], device=th.device(device)).float()).shape[1:])

            flattened_dim = n_flatten + self.n_actions
        else:
            if flatten_input:
                flattened_dim = np.prod(observation_space.shape) + self.n_actions

        if flatten_input:
            self.target_net.append(th.nn.Linear(flattened_dim, hidden_dims[0]))
        else:
            # input already flat
            self.target_net.append(th.nn.Linear(observation_space.shape[0], hidden_dims[0]))

        self.target_net.append(activation())
        for i in range(len(hidden_dims) - 1):
            self.target_net.append(th.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.target_net.append(activation())
        self.target_net.append(th.nn.Linear(hidden_dims[-1], embed_dim))
        self.target_net = th.nn.Sequential(*self.target_net).to(th.device(device))

        self.predict_net = []
        if flatten_input:
            self.predict_net.append(th.nn.Linear(flattened_dim, hidden_dims[0]))
        else:
            # input already flat
            self.predict_net.append(th.nn.Linear(observation_space.shape[0], hidden_dims[0]))
        self.predict_net.append(activation())
        for i in range(len(hidden_dims) - 1):
            self.predict_net.append(th.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.predict_net.append(activation())
        self.predict_net.append(th.nn.Linear(hidden_dims[-1], embed_dim))
        self.predict_net = th.nn.Sequential(*self.predict_net).to(th.device(device))

        if self.use_resnet:
            self.optimizer = th.optim.Adam(list(self.predict_net.parameters()) + list(self.predict_cnn.parameters()), lr=learning_rate)
        else:
            self.optimizer = th.optim.Adam(self.predict_net.parameters(), lr=learning_rate)

    def error(self, state, action):
        """Computes the error between the prediction and target network."""
        if not isinstance(state, th.Tensor):
            state = th.as_tensor(state, device=self.device)
            action = th.as_tensor(action, device=self.device)
        if len(state.shape) == 1:
            # need to add batch dimension to flat input
            state = state.unsqueeze(dim=0)
        if len(state.shape) == 3:
            # need to add batch dimension to image input
            state = state.unsqueeze(dim=0)

        if len(action.shape) == 2:
            # it has an unnecessary dimension that we want to squeeze
            action = action.squeeze(dim=-1)
        if len(action.shape) == 0:
            # need to add a dimension
            action = action.unsqueeze(dim=0)

        if self.normalize_images:
            state = state / 255.
        onehot_action =  th.nn.functional.one_hot(action.long(), num_classes=self.n_actions).float()

        if self.use_resnet:
            x_predict = self.predict_cnn(state)
            x_predict = th.flatten(x_predict, start_dim=1)
            x_target = self.target_cnn(state)
            x_target = th.flatten(x_target, start_dim=1)
        else:
            x_predict = th.flatten(state, start_dim=1)
            x_target = th.flatten(state, start_dim=1)

        x_predict = th.concat([x_predict,  onehot_action], dim=-1)
        x_target = th.concat([x_target,  onehot_action], dim=-1)

        return self.criterion(self.predict_net(x_predict), self.target_net(x_target))

    def observe(self, state, action):
        """Observes state(s) and 'remembers' them using Random Network Distillation"""
        self.optimizer.zero_grad()
        loss = self.error(state, action).mean()
        loss.backward()
        self.optimizer.step()

    def __call__(self, state, action, update_rms=False):
        """Returns the estimated uncertainty for observing a (minibatch of) state(s) as Tensor."""
        rnd = self.error(state, action).mean(dim=-1)

        if update_rms and self.normalize_output:
            self.rnd_rms.update(rnd.cpu().numpy())

        if self.normalize_output:
            rnd = rnd / th.sqrt(th.as_tensor(self.rnd_rms.var, device=self.device) + self.norm_epsilon)

        return rnd
