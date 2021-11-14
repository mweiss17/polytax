import torch


class IterableDataset(torch.utils.data.IterableDataset):
    # TODO: Clean this up https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#IterableDataset
    def __init__(self, seqiotask):
        self.seqiotask = seqiotask

    def __iter__(self):
        def process_sample(sample):
            labels = torch.tensor(sample["decoder_target_tokens"], dtype=torch.long)
            sample = {
                "input_ids": torch.tensor(
                    sample["encoder_input_tokens"], dtype=torch.long
                ),
                "attention_mask": torch.tensor(
                    sample["encoder_segment_ids"], dtype=torch.long
                ),
                "decoder_input_ids": torch.tensor(
                    sample["decoder_input_tokens"], dtype=torch.long
                ),
                "decoder_attention_mask": torch.tensor(
                    sample["decoder_loss_weights"], dtype=torch.long
                ),
                "labels": torch.where(labels >= 32000, -100, labels),
            }
            return sample

        return map(process_sample, self.seqiotask,)


class MapDataset(torch.utils.data.Dataset):
    # TODO: Clean this up https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#IterableDataset
    def __init__(self, ds):
        self.data = list(ds)

    def __getitem__(self, idx):
        labels = torch.tensor(self.data[idx]["decoder_target_tokens"], dtype=torch.long)
        sample = {
            "input_ids": torch.tensor(
                [self.data[idx]["encoder_input_tokens"]], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [self.data[idx]["encoder_segment_ids"]], dtype=torch.long
            ),
            "decoder_input_ids": torch.tensor(
                [self.data[idx]["decoder_input_tokens"]], dtype=torch.long
            ),
            "decoder_attention_mask": torch.tensor(
                [self.data[idx]["decoder_loss_weights"]], dtype=torch.long
            ),
            "labels": torch.where(labels >= 32000, -100, labels).unsqueeze(0),
        }

        return sample

    def __len__(self):
        return len(self.data)
