import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn

@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    print("Configuration:\n", cfg)

    # Simple neural network using config values
    model = nn.Sequential(
        nn.Linear(cfg.input_dim, cfg.hidden_dim),
        nn.ReLU(),
        nn.Linear(cfg.hidden_dim, cfg.output_dim)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    print(f"Training for {cfg.epochs} epochs...")

if __name__ == "__main__":
    main()