import random
from typing import Dict, Mapping, Sequence, Tuple, Type, Any, Optional

import numpy as np
import torch
import torch.optim
import torch.optim.optimizer

from deep_mcts.convolutionalnet import ConvolutionalNet
from deep_mcts.game import CellState, Player, GameManager
from deep_mcts.gamenet import GameNet
from deep_mcts.hex.game import HexAction, HexManager, HexState


class ConvolutionalHexNet(GameNet[HexState, HexAction]):
    grid_size: int
    num_residual: int
    channels: int

    def __init__(
        self,
        grid_size: int,
        manager: Optional[GameManager[HexState, HexAction]] = None,
        optimizer_cls: Type["torch.optim.optimizer.Optimizer"] = torch.optim.SGD,
        optimizer_args: Tuple[Any, ...] = (),
        optimizer_kwargs: Mapping[str, Any] = {"lr": 0.01, "momentum": 0.9},
        num_residual: int = 1,
        channels: int = 128,
    ) -> None:
        super().__init__(
            ConvolutionalNet(
                num_residual=num_residual,
                grid_size=grid_size,
                in_channels=3,
                channels=channels,
                policy_features=grid_size ** 2,
                policy_shape=(grid_size, grid_size),
            ),
            manager if manager is not None else HexManager(grid_size),
            optimizer_cls,
            optimizer_args,
            optimizer_kwargs,
        )
        self.grid_size = grid_size
        self.num_residual = num_residual
        self.channels = channels

    def _mask_illegal_moves(
        self, states: Sequence[HexState], output: torch.Tensor
    ) -> torch.Tensor:
        states = torch.stack(
            [
                torch.tensor(state.grid)
                if state.player == Player.FIRST
                else torch.tensor(state.grid).t()
                for state in states
            ]
        )
        legal_moves = (states == CellState.EMPTY).to(
            dtype=torch.float32, device=self.device
        )
        assert legal_moves.shape == output.shape
        result = output * legal_moves
        assert result.shape == output.shape
        return result

    # Since we flip the board for player 0, we need to flip it back
    def forward(self, state: HexState) -> Tuple[float, torch.Tensor]:
        value, action_probabilities = super().forward(state)
        if state.player == Player.SECOND:
            action_probabilities = torch.transpose(action_probabilities, 1, 2)
        return value, action_probabilities

    def sampling_policy(self, state: HexState) -> HexAction:
        _, action_probabilities = self.forward(state)
        action_probabilities = action_probabilities[0, :, :]
        assert action_probabilities.shape == (self.grid_size, self.grid_size)
        action = np.random.choice(self.grid_size ** 2, p=action_probabilities.flatten())
        y, x = np.unravel_index(action, action_probabilities.shape)
        action = HexAction((x, y))
        assert action in self.manager.legal_actions(state)
        return action

    def greedy_policy(self, state: HexState, epsilon: float = 0) -> HexAction:
        if epsilon > 0:
            p = random.random()
            if p < epsilon:
                return random.choice(self.manager.legal_actions(state))
        _, action_probabilities = self.forward(state)
        action_probabilities = action_probabilities[0, :, :]
        assert action_probabilities.shape == (self.grid_size, self.grid_size)
        y, x = np.unravel_index(
            torch.argmax(action_probabilities), action_probabilities.shape
        )
        action = HexAction((x, y))
        assert action in self.manager.legal_actions(state)
        return action

    def evaluate_state(self, state: HexState) -> Tuple[float, Dict[HexAction, float]]:
        value, probabilities = self.forward(state)
        actions = {
            HexAction((x, y)): probabilities[0, y, x].item()
            for y in range(self.grid_size)
            for x in range(self.grid_size)
        }
        if __debug__:
            legal_actions = set(self.manager.legal_actions(state))
            assert all(
                action in legal_actions
                for action, probability in actions.items()
                if probability != 0
            )
        return value, actions

    def state_to_tensor(self, state: HexState) -> torch.Tensor:
        return self.states_to_tensor([state])

    def states_to_tensor(self, states: Sequence[HexState]) -> torch.Tensor:
        players = torch.stack(
            [
                torch.full((self.grid_size, self.grid_size), fill_value=state.player)
                for state in states
            ]
        )
        # We want everything to be from the perspective of the current player.
        # We also want a consistent orientation, the current player's goal
        # should always be connecting north-south. This means we need to flip
        # the board for player 0.
        grids = torch.stack(
            [
                torch.tensor(state.grid)
                if state.player == Player.FIRST
                else torch.tensor(state.grid).t()
                for state in states
            ]
        )
        current_player = (
            grids
            == torch.tensor([state.player for state in states]).reshape((-1, 1, 1))
        ).float()
        other_player = (
            grids
            == torch.tensor([state.player.opposite() for state in states]).reshape(
                (-1, 1, 1)
            )
        ).float()
        #  assert np.all((first_player.sum(axis=1) - second_player.sum(axis=1)) <= 1)
        tensor = torch.stack((current_player, other_player, players), dim=1)
        assert tensor.shape == (len(states), 3, self.grid_size, self.grid_size)
        return tensor

    def distributions_to_tensor(
        self,
        states: Sequence[HexState],
        distributions: Sequence[Mapping[HexAction, float]],
    ) -> torch.Tensor:
        targets = torch.zeros(
            len(distributions), self.grid_size, self.grid_size
        ).float()
        for i, (state, distribution) in enumerate(zip(states, distributions)):
            for action, probability in distribution.items():
                x, y = action.coordinate
                targets[i][y][x] = probability
            # Since we flip the board for player 0, we also need to flip the targets
            if state.player == Player.SECOND:
                targets[i] = targets[i].t()
        assert targets.shape == (len(distributions), self.grid_size, self.grid_size)
        return targets

    def copy(self) -> "ConvolutionalHexNet":
        net = ConvolutionalHexNet(
            self.grid_size,
            self.manager,
            self.optimizer_cls,
            self.optimizer_args,
            self.optimizer_kwargs,
            self.num_residual,
            self.channels,
        )
        net.net.load_state_dict(self.net.state_dict())
        return net
