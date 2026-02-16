# ISL-MARL-Translator

Real-time Indian Sign Language translator using Multi-Agent Reinforcement 
Learning. Recognizes two signs ("Sorry" and "Thank You") using a 
multi-agent architecture with adaptive confidence handling.

## Architecture
Camera Frame → MediaPipe Landmarks (126-dim)
│
┌─────────┴─────────┐
│ │
Shape Agent Motion Agent
(MLP) (LSTM)
Q: [a, b] Q: [c, d]
│ │
└─────────┬─────────┘
│
VDN Mixing
Q_joint = Q1 + Q2
│
DQN Coordinator
[Sorry, ThankYou, WAIT]
│
OUTPUT
"I'm sorry"

text


## Components

| Component | Role | Architecture |
|-----------|------|-------------|
| Shape Agent | Static hand pose analysis | MLP (126→128→64→2) |
| Motion Agent | Temporal trajectory analysis | LSTM (126→64×2→2) |
| VDN Mixer | Combines agent Q-values | Additive decomposition |
| DQN Coordinator | Final decision with WAIT option | MLP (6→32→16→3) |

## Setup

```bash
git clone https://github.com/yourusername/ISL-MARL-Translator.git
cd ISL-MARL-Translator
pip install -r requirements.txt
Usage
Google Colab
Open notebooks/ISL_MARL_Pipeline.ipynb in Google Colab and run all cells.

Local
Bash

python -m src.data_pipeline    # Process images and build dataset
python -m src.agents           # Train Shape + Motion agents
python -m src.dqn              # Train DQN coordinator
python -m src.translator       # Run inference
Data
Place ISL gesture images in:

data/raw/sorry/ (11 images)
data/raw/thankyou/ (10 images)
The pipeline augments these into training sequences using
synthetic temporal jitter and spatial augmentation.

Results
Metric	Value
Shape Agent Accuracy	~85-90%
Motion Agent Accuracy	~80-85%
Combined (MARL) Accuracy	~92-95%
