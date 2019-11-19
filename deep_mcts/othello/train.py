import datetime
from pathlib import Path

from deep_mcts.othello.convolutionalnet import ConvolutionalOthelloNet
from deep_mcts.othello.game import OthelloManager, OthelloState
from deep_mcts.train import train, TrainingConfiguration

if __name__ == "__main__":
    grid_size = 6
    manager = OthelloManager(grid_size)
    anet = ConvolutionalOthelloNet(grid_size, manager)
    save_dir = (
        Path(__file__).resolve().parent
        / "saves"
        / datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    )
    save_dir.mkdir()
    train(
        anet,
        TrainingConfiguration[OthelloState](
            num_games=5000,
            num_simulations=25,
            save_interval=10_000,
            evaluation_interval=10_000,
            save_dir=str(save_dir),
            sample_move_cutoff=10,
            dirichlet_alpha=1,
        ),
    )
