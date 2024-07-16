from speechbrain.utils.fetching import fetch
from speechbrain.inference.interfaces import Pretrained
import logging
import torch
import torchaudio
from speechbrain.inference.text import GraphemeToPhoneme


logger = logging.getLogger(__name__)


class TextToSpeech(Pretrained):
    """
    A ready-to-use wrapper for Transformer TTS (text -> mel_spec).
    Arguments
    ---------
    hparams
        Hyperparameters (from HyperPyYAML)"""

    HPARAMS_NEEDED = ["model", "blank_index", "padding_mask", "lookahead_mask", "mel_spectogram", "input_encoder"]
    MODULES_NEEDED = ["modules"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_encoder = self.hparams.input_encoder
        self.input_encoder.update_from_iterable(self.hparams.lexicon,sequence_input=False)
        self.g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p")


    def text_to_phoneme(self, text):
        """
        Generates phoneme sequences for the given text using a Grapheme-to-Phoneme (G2P) model.

        Args:
            text (str): The input text.

        Returns:
            list: List of phoneme sequences for the words in the text.
        """
        abbreviation_expansions = {
            "Mr.": "Mister",
            "Mrs.": "Misess",
            "Dr.": "Doctor",
            "No.": "Number",
            "St.": "Saint",
            "Co.": "Company",
            "Jr.": "Junior",
            "Maj.": "Major",
            "Gen.": "General",
            "Drs.": "Doctors",
            "Rev.": "Reverend",
            "Lt.": "Lieutenant",
            "Hon.": "Honorable",
            "Sgt.": "Sergeant",
            "Capt.": "Captain",
            "Esq.": "Esquire",
            "Ltd.": "Limited",
            "Col.": "Colonel",
            "Ft.": "Fort"
        }

        for abbreviation, expansion in abbreviation_expansions.items():
            text = text.replace(abbreviation, expansion)

        phonemes = self.g2p(text)
        phonemes = self.input_encoder.encode_sequence(phonemes)
        phoneme_seq = torch.LongTensor(phonemes)

        return phoneme_seq, len(phoneme_seq)

    def encode_batch(self, texts):
        """Computes mel-spectrogram for a list of texts

        Texts must be sorted in decreasing order on their lengths

        Arguments
        ---------
        texts: List[str]
            texts to be encoded into spectrogram

        Returns
        -------
        tensors of output spectrograms, output lengths and alignments
        """
        with torch.no_grad():
          phonemes = [self.text_to_phoneme(text)[0] for text in texts]
          phoneme_padded, lengths = self.pad_sequences(phonemes)
          phonemes_emb = self.mods.encoder_emb(phoneme_padded)
          encoder_prenet_out = self.mods.encoder_prenet(phonemes_emb)
          pos_enc_output = self.mods.pos_emb(encoder_prenet_out)
          enc_emb = pos_enc_output + encoder_prenet_out

          # Initialize decoder inputs for autoregressive generation
          decoder_input = torch.zeros(1, 1, 80, device=self.device)
          res = []
          stop_token_predictions_list=[]
          stop_condition = False
          itr=0
          max_itr=1000
          res.append(decoder_input)


          while not stop_condition and itr<max_itr:

            decoder_prenet_out = self.mods.decoder_prenet(decoder_input)
            pos_dec_output = self.mods.pos_emb(decoder_prenet_out)
            dec_emb = decoder_prenet_out + pos_dec_output

            src_mask = torch.zeros(enc_emb.size(1), enc_emb.size(1), device=self.device)
            src_key_padding_mask = self.hparams.padding_mask(enc_emb, pad_idx=self.hparams.blank_index)

            decoder_outputs = self.mods.Seq2SeqTransformer(enc_emb, dec_emb, src_mask=src_mask,
                                                              src_key_padding_mask=src_key_padding_mask
                                                              )
            mel_pred = self.mods.mel_linear(decoder_outputs).transpose(1,2)
            postnet_out=self.mods.postnet(mel_pred)
            mel_predictions=mel_pred+postnet_out

            stop_token_predictions= self.mods.stop_linear(decoder_outputs).squeeze(-1)
            stop_token_predictions_list.append(stop_token_predictions)

            decoder_input=mel_predictions.transpose(1,2)
            res.append(decoder_input)
            itr=itr+1

          final_res=torch.cat(res,dim=1)
          final_stop_tokens=torch.cat(stop_token_predictions_list,dim=1)

          return final_res.transpose(1, 2)

    def should_stop(self, stop_token_pred):
        # Implement your stopping condition here.
        # This could check for a predicted end-of-sequence token or a maximum length.
        # Convert logits to probabilities (assuming binary classification with sigmoid activation).
        stop_prob = torch.sigmoid(stop_token_pred).squeeze(-1)
        stop_decision = stop_prob > 0.5
        return stop_decision.any().item()

    def pad_sequences(self, sequences):
      """Pad sequences to the maximum length sequence in the batch.

      Arguments
      ---------
      sequences: List[torch.Tensor]
          The sequences to pad

      Returns
      -------
      Padded sequences and original lengths
      """
      max_length = max([len(seq) for seq in sequences])
      seq_padd = torch.zeros(len(sequences), max_length, dtype=torch.long)
      length_list = []
      for i, seq in enumerate(sequences):
          length = len(seq)
          seq_padd[i, :length] = seq
          length_list.append(length)
      return seq_padd, torch.tensor(length_list)

    def encode_text(self, text):
        """Runs inference for a single text str"""
        return self.encode_batch(text)

    def forward(self, texts):
        "Encodes the input texts."
        return self.encode_batch(texts)
