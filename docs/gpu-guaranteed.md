# Guaranteed GPUs

`GUARANTEED_GPU_UUIDS` names the cards the gateway must always be able to start on. It goes in `gateway.env`, as UUIDs from `nvidia-smi --query-gpu=index,uuid --format=csv`:

```
GUARANTEED_GPU_UUIDS=GPU-xxxx,GPU-yyyy
```

Unset (the default), model_manager never stops a process it did not start.

## What it does

Before a primary model (not a co-resident one such as ASR) spawns on a listed card, every compute process there that belongs to no running backend of this model_manager gets SIGTERM, then SIGKILL after `GPU_CLEAR_TERM_SECONDS` (20). Other users' processes are signalled with `sudo -n kill`. Each one is logged with its PID, user and command.

- Listed cards are chosen first for a new instance, and being full does not make a scale-out skip them.
- If a process is still on the card `GPU_CLEAR_KILL_SECONDS` (15) after SIGKILL, the spawn fails and the log names it. A card that could not be cleared, or whose start failed, loses its preference for an hour. A supervised service (docker, systemd) has to be stopped by its owner.
- Idle unload is unchanged. While a card is unloaded others can use it; the next spawn stops them.
- An engine of ours that a restarted model_manager did not adopt (it did not answer `/health` in time) counts as foreign and is stopped by the next start on its card.
- Do not list a card served through an MPS daemon: stopping it would cut off clients on other cards.
- Anything else you run on a listed card, including your own services, is stopped too.

## Hosts

| Host | Cards (nvidia-smi index) | `GUARANTEED_GPU_UUIDS` |
|---|---|---|
| A100 (`a100-1`) | 1, 2, 3 | `GPU-233f80c8-31c4-b7d7-0c33-b827cc4a79ac,GPU-24af1ff5-eaa7-1d6f-318e-ad3907e93ede,GPU-47b1ec6d-2226-64be-614d-494520d05299` |
| gpu4 | 0 | `GPU-391cbb07-44a9-b69a-4b43-828f06a9063e` |

## GPU indices can move

Pins (`MODEL_GPU_IDS`) and slot ids are nvidia-smi indices, and those shift when a card appears or disappears. The A100 had 7 cards when model_manager started on 2026-11-23 and 8 on 2026-11-26 (new one at bus `01:00`, no reboot), which moved every other card up by one. Guaranteed cards are named by UUID and looked up at the moment of clearing, so clearing never follows a moved index; the pins do. After the card count changes, check the pins and restart model_manager.

## Deploying to the A100: renumber the pins

With 8 cards, the cards that were `0,1,2,6` are `1,2,3,7`, and TTS's `5` is `6`. model_manager also starts engines with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so CUDA counts as nvidia-smi does. In the same deployment:

1. In `/etc/llm-model-manager-cn-27b-scaleout.env`, set `MODEL_GPU_IDS="qwen3.8-27b=1,2,3,7;qwen3-tts-1.7b-customvoice=6"`, add `GUARANTEED_GPU_UUIDS` and the host's image settings (the table in `vision.md`).
2. Stop model_manager **and every running engine** before starting it again. An engine adopted under the old numbering would be recorded on the wrong card.
3. Check each instance with `nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv`.
