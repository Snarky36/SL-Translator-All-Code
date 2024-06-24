import random
import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, use_lstm=True, use_attention=False):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.adaptive_ratio = 0
        self.total_epochs = 0
        self.adaptive_teacher = False
        self.nr_epochs_full_teaching = 0
        self.teacher_help_epochs_percent = 1
        self.use_lstm = use_lstm
        self.use_attention = use_attention

    def forward(self, source, target, teacher_force_ratio=0.5, current_epoch=0):

        batch_size = source.shape[1]
        target_len = target.shape[0]

        outputs = torch.zeros(target_len, batch_size, self.decoder.output_size).to(self.device)

        if self.use_lstm:
            if self.use_attention:
                encoder_results, (hidden, cell) = self.encoder(source, use_attention=True)
            else:
                hidden, cell = self.encoder(source)
        else:
            hidden = self.encoder(source)
        # Grab the first input to the Decoder which will be <SOS> token
        next_input = target[0]

        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            if self.use_lstm:
                if self.use_attention:
                    output, hidden, cell = self.decoder(next_input, hidden, cell, encoder_results)
                else:
                    output, hidden, cell = self.decoder(next_input, hidden, cell)
            else:
                output, hidden = self.decoder(next_input, hidden)

            # Store next output prediction
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to. This was a long comment.

            if self.adaptive_teacher:
                if current_epoch < self.nr_epochs_full_teaching:
                    self.calculate_teacher_forcing(current_epoch)
                    teacher_force_ratio = 1.0
                    #print("[MODEL] - current_epoch", current_epoch, " self.nr_epochs_full_teaching", self.nr_epochs_full_teaching)
                else:
                    teacher_force_ratio = self.calculate_teacher_forcing(current_epoch)

            next_input = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

    def get_device(self):
        return self.device

    def setup_adaptive_teacher_forcing(self,adaptive_teacher,
                                       adaptive_ratio=0,
                                       total_epochs=0,
                                       nr_epochs_full_teaching=0,
                                       teacher_help_epoch_percent=1):
        self.adaptive_teacher = adaptive_teacher
        self.adaptive_ratio = adaptive_ratio
        self.total_epochs = total_epochs
        self.nr_epochs_full_teaching = nr_epochs_full_teaching
        self.teacher_help_epochs_percent = teacher_help_epoch_percent


    def calculate_teacher_forcing(self, current_epoch):
        target_epoch = int(self.teacher_help_epochs_percent * self.total_epochs)
        teacher_forcing_value = 1 - (current_epoch / target_epoch * self.adaptive_ratio)
        #print(f"Teacher Forcing Value at epoch {current_epoch}: {teacher_forcing_value:.4f}")
        return max(0, teacher_forcing_value)
