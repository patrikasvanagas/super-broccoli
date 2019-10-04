import random
from typing import Tuple, Dict, Sequence, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from deep_mcts.gamenet import GameNet, DEVICE
from deep_mcts.tictactoe.game import TicTacToeState, TicTacToeAction, TicTacToeManager


class FullyConnectedTicTacToeModule(nn.Module):  # type: ignore
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3 ** 2 * 2 + 1, 128)
        self.value_head = nn.Linear(128, 1)
        self.policy_head = nn.Linear(128, 3 ** 2)

    def forward(  # type: ignore
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = x
        assert input.shape == (input.shape[0], 3 ** 2 * 2 + 1)
        x = F.relu(self.fc1(x))
        policy = self.policy_head(x)
        assert policy.shape == (input.shape[0], (input.shape[1] - 1) / 2)
        value = self.value_head(x)
        assert value.shape == (input.shape[0], 1)
        return value, policy


class FullyConnectedTicTacToeNet(GameNet[TicTacToeState, TicTacToeAction]):
    grid_size: int
    manager: TicTacToeManager

    def __init__(self) -> None:
        super().__init__()
        self.net = FullyConnectedTicTacToeModule().to(DEVICE)
        self.manager = TicTacToeManager()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)

    def _mask_illegal_moves(
        self, states: Sequence[TicTacToeState], output: torch.Tensor
    ) -> torch.Tensor:
        states = np.stack([state.grid for state in states], axis=0).reshape(-1, 9)
        legal_moves = torch.as_tensor(states == -1, dtype=torch.float32, device=DEVICE)
        assert legal_moves.shape == output.shape
        result = output * legal_moves
        assert result.shape == output.shape
        return result

    def sampling_policy(self, state: TicTacToeState) -> TicTacToeAction:
        _, action_probabilities = self.forward(state)
        action_probabilities = action_probabilities[0, :]
        assert action_probabilities.shape == (3 ** 2,)
        action = np.random.choice(len(action_probabilities), p=action_probabilities)
        x, y = action % 3, action // 3
        return TicTacToeAction((x, y))

    def greedy_policy(
        self, state: TicTacToeState, epsilon: float = 0
    ) -> TicTacToeAction:
        if epsilon > 0:
            p = random.random()
            if p < epsilon:
                return random.choice(self.manager.legal_actions(state))
        _, action_probabilities = self.forward(state)
        action_probabilities = action_probabilities[0, :]
        assert action_probabilities.shape == (3 ** 2,)
        action = np.argmax(action_probabilities)
        x, y = action % 3, action // 3
        return TicTacToeAction((x, y))

    def evaluate_state(
        self, state: TicTacToeState
    ) -> Tuple[float, Dict[TicTacToeAction, float]]:
        value, probabilities = self.forward(state)
        probabilities = probabilities[0, :]
        assert probabilities.shape == (3 ** 2,)
        actions = {
            TicTacToeAction((x, y)): probabilities[y * 3 + x]
            for y in range(3)
            for x in range(3)
        }
        return value, actions

    def _state_to_tensor(self, state: TicTacToeState) -> torch.Tensor:
        # grid = state.grid
        # tensor = [0] * (3 ** 2 * 2 + 1)
        # tensor[-1] = state.player
        # for y in range(3):
        #     for x in range(3):
        #         # TODO: Make current player come first instead of always player 0?
        #         tensor[y * 3 + x] = 1 if grid[y][x] == 0 else 0
        #         tensor[3 ** 2 + y * 3 + x] = 1 if grid[y][x] == 1 else 0
        # tensor = torch.tensor(tensor).reshape(1, -1)
        # assert tensor.shape == (1, 2 * 3 ** 2 + 1)
        # return tensor
        return self._states_to_tensor([state])

    def _states_to_tensor(self, states: Sequence[TicTacToeState]) -> torch.Tensor:
        players = np.array([state.player for state in states]).reshape(
            (len(states), -1)
        )
        grids = np.stack([state.grid for state in states]).reshape((len(states), -1))
        for i in range(len(states)):
            assert np.all(grids[i, :] == np.array(states[i].grid).flatten())
        first_player = grids == 0
        second_player = grids == 1
        assert np.all((first_player.sum(axis=1) - second_player.sum(axis=1)) <= 1)
        # TODO: Make current player come first instead of always player 0?
        states = np.concatenate((first_player, second_player, players), axis=1)
        tensor = torch.as_tensor(states, dtype=torch.float32, device=DEVICE)
        assert tensor.shape == (len(states), 2 * 3 ** 2 + 1)
        return tensor
        # players = np.array([state.player for state in states]).reshape(
        #     (len(states), -1)
        # )
        # grids = np.stack([state.grid for state in states]).reshape((len(states), -1))
        # for i in range(len(states)):
        #     assert np.all(grids[i, :] == np.array(states[i].grid).flatten())
        # current_player = grids == np.array([state.player for state in states]).reshape(
        #     (-1, 1)
        # )
        # other_player = grids == np.array(
        #     [0 if state.player == 1 else 1 for state in states]
        # ).reshape((-1, 1))
        # assert np.all((current_player.sum(axis=1) - other_player.sum(axis=1)) <= 1)
        # grids = np.concatenate((current_player, other_player, players), axis=1)
        # tensor = torch.as_tensor(grids, dtype=torch.float32, device=DEVICE)
        # assert tensor.shape == (len(grids), 2 * 3 ** 2 + 1)
        # return tensor

    def _distributions_to_tensor(
        self,
        states: Sequence[TicTacToeState],
        distributions: Sequence[Mapping[TicTacToeAction, float]],
    ) -> torch.Tensor:
        targets = np.zeros((len(distributions), 3 ** 2), dtype=np.float32)
        for i, distribution in enumerate(distributions):
            for action, probability in distribution.items():
                x, y = action.coordinate
                targets[i][y * 3 + x] = probability
        targets = torch.as_tensor(targets, device=DEVICE)
        assert targets.shape == (len(distributions), 3 ** 2)
        return targets

    def copy(self) -> "FullyConnectedTicTacToeNet":
        net = FullyConnectedTicTacToeNet()
        net.net.load_state_dict(self.net.state_dict())
        return net