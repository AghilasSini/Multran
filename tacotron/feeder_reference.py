from tacotron.feeder import Feeder
import numpy as np
from tacotron.utils.text import text_to_sequence, ipa_to_articulatory_sequence
import os

class FeederReference(Feeder):

    def __init__(self, coordinator, metadata_filename, hparams, training=True, split=True):
        super().__init__(coordinator, metadata_filename, hparams, training=training, split=split)

    def _get_next_example(self):
        """Gets a single example (input, mel_target, token_target, linear_target, mel_length) from_ disk
        """
        if self._train_offset >= len(self._train_meta):
            self._train_offset = 0
            np.random.shuffle(self._train_meta)

        meta = self._train_meta[self._train_offset]
        self._train_offset += 1

        if not self._hparams.tacotron_phoneme_transcription:
            text = meta[5]
            input_data = np.asarray(text_to_sequence(text, self._cleaner_names), dtype=np.int32)

        # Phoneme transcription
        else:
            '''
            text_as_words = meta[5].split(' ')
            text_as_phonemes = meta[6].split(' ')
            assert len(text_as_words) == len(text_as_phonemes)
            for i in range(0, len(text_as_words)):
                random_number = np.random.random()
                if random_number < self._proba_phoneme:
                    text_as_words[i] = text_as_phonemes[i]
            text = " ".join(text_as_words)
            '''
            text = meta[6]
            input_data = np.asarray(ipa_to_articulatory_sequence(text), dtype=np.int32)

        if self._hparams.tacotron_multi_speaker:
            speaker_id = [0 for i in range(int(self._nb_speaker))]
            speaker_id[int(meta[7])] = 1
        print()
        mel_reference = np.load(os.path.join(self._mel_dir, meta[2]))

        # input_data = np.asarray(text_to_sequence(text, self._cleaner_names), dtype=np.int32)
        mel_target = np.load(os.path.join(self._mel_dir, meta[8]))
        # Create parallel sequences containing zeros to represent a non finished sequence
        token_target = np.asarray([0.] * (len(mel_target) - 1))
        linear_target = np.load(os.path.join(self._linear_dir, meta[2]))

        if self._hparams.tacotron_multi_speaker:
            return (input_data, mel_target, token_target, linear_target, len(mel_target), speaker_id, mel_reference)
        else:
            return (input_data, mel_target, token_target, linear_target, len(mel_target), mel_reference)

    def _prepare_batch(self, batches, outputs_per_step):

        if self._hparams.tacotron_multi_speaker:
            (inputs, input_lengths, mel_targets, token_targets, linear_targets, targets_lengths, split_infos,
             mel_refs, speaker_ids) = super()._prepare_batch(batches, outputs_per_step)
        else:
            (inputs, input_lengths, mel_targets, token_targets, linear_targets, targets_lengths, split_infos,
             mel_refs) = super()._prepare_batch(batches, outputs_per_step)

        size_per_device = int(len(batches) / self._hparams.tacotron_num_gpus)

        mel_refs = None
        for i in range(self._hparams.tacotron_num_gpus):
            batch = batches[size_per_device * i:size_per_device * (i + 1)]
            mel_ref_cur_device, mel_ref_max_len = self._prepare_targets([x[-1] for x in batch], outputs_per_step)
            mel_refs = np.concatenate((mel_refs, mel_ref_cur_device), axis=1) if mel_refs is not None else mel_ref_cur_device

        if self._hparams.tacotron_multi_speaker:
            return (inputs, input_lengths, mel_targets, token_targets, linear_targets, targets_lengths, split_infos,
                    mel_refs, speaker_ids)
        else:
            return (inputs, input_lengths, mel_targets, token_targets, linear_targets, targets_lengths, split_infos,
                    mel_refs)
