from typing import Mapping, Sequence, Tuple, Type, Any, Optional, Dict

import torch
import torch.optim
import torch.optim.optimizer

from deep_mcts.convolutionalnet import ConvolutionalNet
from deep_mcts.game import CellState, Player, GameManager
from deep_mcts.gamenet import GameNet
from deep_mcts.hex.game import Action, HexManager, HexState


class ConvolutionalHexNet(GameNet[HexState]):
    grid_size: int
    num_residual: int
    channels: int
    value_head_hidden_units: int

    def __init__(
        self,
        grid_size: int,
        manager: Optional[GameManager[HexState]] = None,
        optimizer_cls: Type["torch.optim.optimizer.Optimizer"] = torch.optim.SGD,
        optimizer_args: Tuple[Any, ...] = (),
        optimizer_kwargs: Mapping[str, Any] = {
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.001,
        },
        num_residual: int = 3,
        channels: int = 128,
        value_head_hidden_units: int = 128,
    ) -> None:
        super().__init__(
            ConvolutionalNet(
                num_residual=num_residual,
                grid_size=grid_size,
                in_channels=3,
                channels=channels,
                value_head_hidden_units=value_head_hidden_units,
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
        self.value_head_hidden_units = value_head_hidden_units

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Since we flip the board for the second player, we need to flip it back
        values, action_probabilities = super().forward(states)
        players = states[:, 2, 0, 0]
        second_player_states = players == Player.SECOND
        action_probabilities[second_player_states, :, :] = action_probabilities[
            second_player_states, :, :
        ].transpose(1, 2)
        return values, action_probabilities

    def states_to_tensor(self, states: Sequence[HexState]) -> torch.Tensor:
        second_player_states, players, opposite_players, grids = zip(
            *[
                (
                    state.player == Player.SECOND,
                    state.player,
                    state.player.opposite(),
                    state.grid,
                )
                for state in states
            ]
        )
        players = torch.tensor(players).reshape(-1, 1, 1)
        opposite_players = torch.tensor(opposite_players).reshape(-1, 1, 1)
        player_grids = torch.full(
            (len(states), self.grid_size, self.grid_size), fill_value=-1.0
        )
        player_grids[:] = players
        # We want everything to be from the perspective of the current player.
        # We also want a consistent orientation, the current player's goal
        # should always be connecting north-south. This means we need to flip
        # the board for the second player.
        grids = torch.tensor(grids)
        grids[second_player_states, :] = grids[second_player_states, :].transpose(1, 2)

        current_player = (grids == players).float()
        other_player = (grids == opposite_players).float()
        #  assert np.all((first_player.sum(axis=1) - second_player.sum(axis=1)) <= 1)
        tensor = torch.stack((current_player, other_player, player_grids), dim=1)
        assert tensor.shape == (len(states), 3, self.grid_size, self.grid_size)
        return tensor

    def copy(self) -> "ConvolutionalHexNet":
        net = ConvolutionalHexNet(
            self.grid_size,
            self.manager,
            self.optimizer_cls,
            self.optimizer_args,
            self.optimizer_kwargs,
            self.num_residual,
            self.channels,
            self.value_head_hidden_units,
        )
        net.load_state_dict(self.net.state_dict())
        return net

    @classmethod
    def from_path_full(
        cls, path: str, manager: Optional[GameManager[HexState]] = None,
    ) -> "ConvolutionalHexNet":
        parameters = torch.load(path, map_location=torch.device("cpu"))
        net = cls(
            grid_size=parameters["grid_size"],
            manager=manager,
            optimizer_cls=parameters["optimizer_cls"],
            optimizer_args=parameters["optimizer_args"],
            optimizer_kwargs=parameters["optimizer_kwargs"],
            num_residual=parameters["num_residual"],
            value_head_hidden_units=parameters["value_head_hidden_units"],
            channels=parameters["channels"],
        )
        net.load_state_dict(parameters["state_dict"])
        return net

    def parameters(self) -> Dict[str, Any]:
        return {
            **super().parameters(),
            "grid_size": self.grid_size,
            "num_residual": self.num_residual,
            "channels": self.channels,
            "value_head_hidden_units": self.value_head_hidden_units,
        }
