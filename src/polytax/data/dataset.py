import torch
import numpy as np

class IterableDataset(torch.utils.data.IterableDataset):
    # TODO: Clean this up https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#IterableDataset
    def __init__(self, np_iterator):
        self.np_iterator = np_iterator

    def __iter__(self):
        def process_sample(sample):
            if sample.get("encoder_segment_ids") is None:
                attention_mask = torch.tensor(sample["encoder_input_tokens"], dtype=torch.long)
                attention_mask = torch.where(attention_mask >= 1, 1, attention_mask)
            else:
                attention_mask = torch.tensor(
                    [sample["encoder_segment_ids"]], dtype=torch.long
                ).squeeze()
            labels = torch.tensor(sample["decoder_target_tokens"], dtype=torch.long)
            sample = {
                "input_ids": torch.tensor(
                    sample["encoder_input_tokens"], dtype=torch.long
                ),
                "attention_mask": attention_mask,
                "decoder_input_ids": torch.tensor(
                    sample["decoder_input_tokens"], dtype=torch.long
                ),
                "decoder_attention_mask": torch.tensor(
                    sample["decoder_loss_weights"], dtype=torch.long
                ),
                "labels": torch.where(labels >= 32000, -100, labels),
            }
            return sample

        return map(process_sample, self.np_iterator,)


class MapDataset(torch.utils.data.Dataset):
    # TODO: Clean this up https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#IterableDataset
    def __init__(self, ds):
        self.data = list(ds)

    def __getitem__(self, idx):
        if self.data[idx].get("encoder_segment_ids") is None:
            attention_mask = torch.tensor(self.data[idx]["encoder_input_tokens"], dtype=torch.long)
            attention_mask = torch.where(attention_mask >= 1, 1, attention_mask)
        else:
            attention_mask = torch.tensor(
                [self.data[idx]["encoder_segment_ids"]], dtype=torch.long
            )

        labels = torch.tensor(self.data[idx]["decoder_target_tokens"], dtype=torch.long)
        labels = torch.where(labels >= 32000, -100, labels).unsqueeze(0)
        sample = {
            "input_ids": torch.tensor(
                [self.data[idx]["encoder_input_tokens"]], dtype=torch.long
            ),
            "decoder_input_ids": torch.tensor(
                [self.data[idx]["decoder_input_tokens"]], dtype=torch.long
            ),
            "decoder_attention_mask": torch.tensor(
                [self.data[idx]["decoder_loss_weights"]], dtype=torch.long
            ),
            "labels": labels,
            "attention_mask": attention_mask
        }

        return sample

    def collate_fn(self, features):
        batch = {"input_ids": torch.stack([feature["input_ids"] for feature in features]).squeeze(),
                 "attention_mask": torch.stack([feature["attention_mask"] for feature in features]).squeeze(),
                 "decoder_input_ids": torch.stack([feature["decoder_input_ids"] for feature in features]).squeeze(),
                 "decoder_attention_mask": torch.stack([feature["decoder_attention_mask"] for feature in features]).squeeze(),
                 "labels": torch.stack([feature["labels"] for feature in features]).squeeze()
                 }
        return batch

    def __len__(self):
        return len(self.data)
