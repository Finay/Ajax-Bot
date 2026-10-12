# Ajax Bot

A reinforcement learning agent for [UTMIST AI² 2025](https://github.com/UTMIST/UTMIST-AI2), a 1v1 platform-fighting tournament, with its policy brought up to run on Tenstorrent hardware through `ttnn`. Winner of the Tenstorrent track.

<img src="docs/aisquaredv4.gif"></img>

This repository is a fork of the tournament starter kit. The game environment, tournament server, CI pipelines and the files under `guides/` come from the UTMIST organizers (see [Credits](#credits)). The work described below is what this fork and its predecessor, [UTMIST-AI2-prev](https://github.com/Finay/UTMIST-AI2-prev), add on top.

## What is here

| Path | What it is |
|---|---|
| `user/my_agent.py` | The submitted agent: PPO policy, custom feature extractor, Tenstorrent inference path |
| `user/ppo_autobringup.py` | `TTPolicy`, a drop-in `ttnn` bring-up for any SB3 PPO `MlpPolicy` |
| `user/AUTOBRINGUP.md` | Guide to `TTPolicy`, written so other teams could use it |
| `user/ppo_to_ttnn.py`, `user/agent_inspector.ipynb` | Prototypes and notes from working out the SB3 policy internals |

## The agent

`SubmittedAgent` in `user/my_agent.py` is a Stable-Baselines3 PPO agent.

- **Feature extractor.** `CustomExtractor` turns the raw observation into 86 features. Continuous values pass through; the categorical fields of both players are one-hot encoded so the network does not have to treat ids as magnitudes.
- **Policy network.** Three hidden layers of 64 units with tanh activations, followed by SB3's linear action head.
- **PPO settings.** Rollouts of 13,500 steps (five 90-second matches at 30 fps), batch size 150, entropy coefficient 0.01.
- **Ledge safeguard.** When the agent stands at the edge of the stage and the opponent is not next to it, every key except jump and taunt is masked, which stops it from walking off.
- **Weights.** The submitted checkpoint was saved at about 47 million timesteps. It is downloaded on first run to `rl-model.zip`.

## Tenstorrent bring-up

`TTPolicy` moves the policy's forward pass onto a Tenstorrent device.

- It reads the weights of SB3's `mlp_extractor.policy_net` and `action_net`, converts them to `ttnn` tensors, and runs the tanh-linear stack and the action head with `ttnn.linear` and `ttnn.tanh`.
- It bypasses SB3's own `predict`, so there is one conversion into `ttnn` and one back out per step, rather than one per layer.
- The feature extractor stays in PyTorch. It is architecture-specific, and only the features-to-action path is generic across `MlpPolicy` agents. The critic is not brought up because inference does not need it.
- **PCC validation.** With `run_pcc_validation=True`, each step compares the `ttnn` output of the policy net and of the action head against PyTorch using the Pearson correlation coefficient (threshold 0.99), then checks the full pass against the deterministic SB3 action.
- **Fallback.** If `ttnn` cannot be imported, the agent runs the same model in PyTorch, so one file works on both the Tenstorrent runner and a laptop.

Using it in another agent takes three steps; see [`user/AUTOBRINGUP.md`](user/AUTOBRINGUP.md).

## Earlier work: UTMIST-AI2-prev

The training experiments that led to this agent live in [UTMIST-AI2-prev](https://github.com/Finay/UTMIST-AI2-prev).

**DQN from scratch** (`main`, `CustomTrainer.ipynb`). A Deep Q-Network written directly in PyTorch, without SB3:

- replay memory, target network with soft updates, epsilon-greedy exploration, Huber loss
- 32 discrete actions decoded into key presses
- a 60-frame stack of processed observations plus the previous action as input to a five-layer, 1024-unit MLP
- shaped rewards for knockouts, proximity, damage, attacking in range, and falling into the danger zone

**PPO training harness** (`fix` and `elo` branches, `train.py`, `custom_agents.py`). The move to PPO:

- a command-line trainer where reward terms and their weights are passed as JSON, with each run's configuration saved beside its checkpoints and logged to TensorBoard
- `PPOAgent` with the one-hot feature extractor used in the final agent, and a `RecurrentPPOAgent` (LSTM) variant
- reward terms for movement, proximity, damage dealt and taken, staying on stage, knockouts and wins

**Elo self-play league** (`elo` branch, `leaderboard_tools.py`, `environment/agent.py`):

- every saved checkpoint plays 16 placement matches and joins `leaderboard.json` with an Elo rating
- during training, each episode's opponent is the leaderboard entry closest to the learner's current rating, so the agent keeps facing opponents near its own strength
- a separate tool replays matches among the top 20 to refine their ratings

Reward shaping and the match runner in that repository were built together with Kumaran Surya.

## Running

Python 3.10. From the repository root:

```bash
pip install -r requirements.txt
python user/demo_match.py        # one match against the scripted BasedAgent
pytest -s user/validate.py       # the tournament's validation check
```

`install.sh` and `install_tt.sh` are the organizers' setup scripts for the CI runners; `install_tt.sh` covers the Tenstorrent environment. Without `ttnn` installed the agent runs in PyTorch.

## Credits

- Environment, tournament server, starter code and guides: [UTMIST/UTMIST-AI2](https://github.com/UTMIST/UTMIST-AI2), by the University of Toronto Machine Intelligence Student Team. See their [Colab notebook](https://colab.research.google.com/drive/1V184vtHSagN13L0SbWGmnY-jCDvIefmm?usp=sharing) and [paper](https://drive.google.com/file/d/1G0hatGPBXvh2j5byjfrqKBthknNOt5sp/view).
- Starter documentation: [`guides/technical_guide.md`](guides/technical_guide.md) and [`guides/tt_guide.md`](guides/tt_guide.md).
