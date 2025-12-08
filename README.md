src/drums_SAE/
├── sae/
│   ├── __init__.py
│   ├── model.py          # SAE architecture with RMSNorm
│   ├── config.py         # Dataclass-based config
│   └── loss.py           # Loss functions (MSE, L1, AuxK)
├── training/
│   ├── __init__.py
│   ├── dataloader.py     # Efficient latent data loading
│   ├── trainer.py        # Training loop with logging
│   └── callbacks.py      # Checkpointing, early stopping
├── utils/
│   └── metrics.py        # Sparsity, explained variance, L0
└── scripts/
    └── train_sae.py      # Entry point
    
    
    
2how to do handle dead feature:
- ghost gradient
 - aux k loss
 - resampling??

Hidden dim: For your 64-dim input → try {256, 512, 1024, 2048, 4096} (4x to 64x)
Sparsity λ: {0.005, 0.01, 0.05, 0.1}
Learning rate: 1e-4 to 3e-4 with warmup
