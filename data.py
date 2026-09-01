# Byte-level data for Wyrmling. Chat lives here, not in the model.
#
# Chat framing uses reserved control bytes that (essentially) never appear in
# clean UTF-8 text, so the byte-level model learns turn structure without a
# tokenizer and without stealing a real character:
#   0x02 STX  -> start of a USER turn
#   0x03 ETX  -> start of an ASSISTANT turn
#   0x04 EOT  -> end of conversation
# A turn is: <role_byte> <utf-8 text bytes>. Conversation ends with EOT.
import numpy as np
import torch

USER = 0x02
ASSISTANT = 0x03
EOT = 0x04
CONTROL_BYTES = (USER, ASSISTANT, EOT)


def encode_conversation(messages):
    """messages: [{"role": "user"|"assistant", "content": str}] -> bytes.

    Returns (ids, mask) as python lists. mask=1 on assistant content + its
    trailing EOT (the bytes we want the model to learn to produce), 0 elsewhere.
    """
    ids, mask = [], []
    for m in messages:
        role = USER if m["role"] == "user" else ASSISTANT
        body = m["content"].encode("utf-8", errors="replace")
        ids.append(role); mask.append(0)                     # role marker: never a target
        ids.extend(body); mask.extend([1 if role == ASSISTANT else 0] * len(body))
    ids.append(EOT); mask.append(1)
    return ids, mask


def encode_text(s: str):
    """Raw text -> byte ids (for pretrain on web/books; no chat markers)."""
    return list(s.encode("utf-8", errors="replace"))


class ByteData:
    """Random-window sampler over a memmapped uint8 .bin (nanoGPT-style).

    Used for pretraining and for packed-chat pretraining. For masked chat
    finetuning use a token+mask .npz via `MaskedChatData` (see prepare_data.py).
    """

    def __init__(self, data_dir, block_size, batch_size, device):
        import os
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.train = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint8, mode="r")
        val_path = os.path.join(data_dir, "val.bin")
        self.val = np.memmap(val_path, dtype=np.uint8, mode="r") if os.path.exists(val_path) else None

    def get_batch(self, split="train"):
        data = self.train if split == "train" or self.val is None else self.val
        bs, T = self.batch_size, self.block_size
        ix = torch.randint(len(data) - T - 1, (bs,))
        x = torch.stack([torch.from_numpy(data[i : i + T].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + T].astype(np.int64)) for i in ix])
        if self.device.type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
