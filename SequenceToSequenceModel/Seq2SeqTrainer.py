import time
from torch import optim, nn
from torch.utils.data import DataLoader, random_split

from AttentionModel.Attention import Attention
from AttentionModel.DecoderAttention import DecoderAttention
from GRU_Decoder import GRUDecoder
from GRU_Encoder import GRUEncoder
from SignLanguageDataset import SignLanguageDataset
from torch.utils.tensorboard import SummaryWriter
import torch
from Encoder import Encoder
from Decoder import Decoder
from Seq2SeqModel import Seq2Seq
from Utils import load_checkpoint, save_checkpoint


model_name = "zaha-german-gloss-27-06-Attention-V1"
prefix = "Translate this text into his gloss form:"
max_input_length = 64
signLanguageDatasets = SignLanguageDataset(prefix=prefix, max_length=max_input_length, path="./dataset/", language="en")

tensor_board_name = input("Tensor Board folder name: ")
writer = SummaryWriter(f"/out/LSTM_Checkpoints/runs/{model_name}-{tensor_board_name}")


def log_training_parameters(writer, params, step=0):
    for key, value in params.items():
        writer.add_text(key, str(value), step)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prepare_Model(
        encoder_size,
        decoder_size,
        encoder_embedding_size=256,
        decoder_embedding_size=256,
        hidden_size=1024,
        num_layers=4,
        encoder_dropout=0.5,
        decoder_dropout=0.5,
        prepare_for_training=True,
        use_lstm=True,
        use_attention=False
        ):
    # Model hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size_encoder = encoder_size
    input_size_decoder = decoder_size
    output_size = decoder_size
    encoder_embedding_size = encoder_embedding_size
    decoder_embedding_size = decoder_embedding_size
    hidden_size = hidden_size  # Needs to be the same for both encoder and decoder?
    num_layers = num_layers
    enc_dropout = encoder_dropout
    dec_dropout = decoder_dropout

    print("num_layers", num_layers)
    print("hidden_size", hidden_size)
    print("use lstm", use_lstm)
    print("use attention", use_attention)
    if use_lstm:
        encoder = Encoder(
            input_size=input_size_encoder,
            embedding_size=encoder_embedding_size,
            hidden_size=hidden_size,
            nr_layers=num_layers,
            dropout=enc_dropout
        )
        if use_attention:
            attention = Attention(encoder_hidden_size=hidden_size, decoder_hidden_size=hidden_size)
            decoder = DecoderAttention(
                    input_size=input_size_decoder,
                    embedding_size=decoder_embedding_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    nr_layers=num_layers,
                    dropout=dec_dropout,
                    attention=attention
            )
        else:
            decoder = Decoder(
                input_size=input_size_decoder,
                embedding_size=decoder_embedding_size,
                hidden_size=hidden_size,
                output_size=output_size,
                nr_layers=num_layers,
                dropout=dec_dropout
            )
    else:
        encoder = GRUEncoder(
            input_size=input_size_encoder,
            embedding_size=encoder_embedding_size,
            hidden_size=hidden_size,
            nr_layers=num_layers,
            dropout=enc_dropout
        )

        decoder = GRUDecoder(
            input_size=input_size_decoder,
            embedding_size=decoder_embedding_size,
            hidden_size=hidden_size,
            output_size=output_size,
            nr_layers=num_layers,
            dropout=dec_dropout
        )
    z_model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        device=device,
        use_lstm=use_lstm,
        use_attention=use_attention
    )

    #resize the embedding number because i added some ne special tokens and i will use new tokens
    #z_model.resize_token_embeddings(len(tokenizer))
    if prepare_for_training:
        z_model.apply(init_weights) #initializez wheigturile poate ajuta la training

    #paralelizez procesele
    if torch.cuda.device_count() > 1:
        z_model = nn.DataParallel(z_model)
    z_model.to(device)

    print(f"The model has {count_parameters(z_model):,} trainable parameters")
    print("Model shape", z_model)

    return z_model


def collate_fn(batch):

    input_seqs = [torch.tensor(example['input_ids']) for example in batch]
    input_lens = torch.tensor([len(seq) for seq in input_seqs])

    max_len = max(input_lens)
    padded_seqs = [torch.nn.functional.pad(seq, (0, max_len - len(seq)), 'constant', 0) for seq in input_seqs]

    padded_tensor = torch.stack(padded_seqs)

    return padded_tensor


def create_optimizer(model, learning_rate, optimizer_choice="Adam", momentum=0.9, lr_decay=0.2):
    if optimizer_choice == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    if optimizer_choice == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    if optimizer_choice == "AdaGrad":
        return optim.Adagrad(model.parameters(), lr=learning_rate, lr_decay=lr_decay)


def set_learning_rate(optimizer, learning_rate, halving_ratio):
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate * halving_ratio


def validation_epoch(model, data_loader, device, criterion):
    epoch_loss = 0
    total_samples_on_epoch = 0
    total_correct = 0
    total_count = 0
    evaluation_steps = len(data_loader)
    start_time = time.time()
    for step, batch in enumerate(data_loader):
        source = batch[0].to(device)
        target = batch[1].to(device)

        source = source.transpose(0, 1)
        target = target.transpose(0, 1)

        total_samples_on_epoch += source.size(0)

        # Forward pass
        output = model(source, target, teacher_force_ratio=0)

        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        loss = criterion(output, target)
        epoch_loss += loss.item()

        predictions = output.argmax(1)
        total_correct += (predictions == target).sum().item()
        total_count += target.size(0)



        print(
            f'Evaluation Step [{step}/{evaluation_steps}] => loss = {loss.item():.3f} || epoch_loss = {epoch_loss / (step + 1):.3f}')
    end_time = time.time()

    validation_runtime = end_time - start_time
    validation_samples_per_second = total_samples_on_epoch / validation_runtime
    accuracy = total_correct / total_count
    return epoch_loss/evaluation_steps, validation_samples_per_second, validation_runtime, accuracy


def train(
        model_name,
        num_epochs=4,
        learning_rate=1e-3,
        batch_size=64,
        load_model=False,
        save_at_nr_steps=300,
        save_at_nr_epoch=None,
        respect_dataset_max_length=False,
        optimizer_choice="Adam",
        teacher_forcing_ratio=0.0,
        teacher_help_percent=0.0,
        adaptive_teacher_forcing=True,
        adaptive_teacher_on_intervals=None,
        scheduler_patience=10,
        scheduler_threshold=0.001,
        scheduler_factor=0.5,
        manual_schedular_decreasing=False,
        schedular_decreasing_epochs=None,
        **model_kwargs):

    tokenized_train, tokenized_test, tokenized_dev = signLanguageDatasets.get_tokenized_datasets()
    train_tensor_dataset, test_tensor_dataset, dev_tensor_dataset = signLanguageDatasets.get_tensor_datasets()
    train_tensor_dataset, validation_tensor_dataset = signLanguageDatasets.split_dataset_and_get_validation(train_tensor_dataset)
    vocab_size = signLanguageDatasets.get_train_vocab_size()

    max_tokens = signLanguageDatasets.get_max_nr_of_tokens(respect_dataset_max_length)

    tokenizer = signLanguageDatasets.get_tokenizer()

    print("[Training ] MaxTokens= ", max_tokens)
    print(f"[Training] Respect Dataset Max length = {respect_dataset_max_length} => input max length= {max_input_length}")

    model = prepare_Model(
        encoder_size=vocab_size,
        decoder_size=vocab_size,
        encoder_embedding_size=max_tokens,
        decoder_embedding_size=max_tokens,
        **model_kwargs
    )
    print(f"Model num epochs = {num_epochs}")
    if torch.cuda.device_count() > 1:
        model.module.setup_adaptive_teacher_forcing(adaptive_teacher=adaptive_teacher_forcing,
                                                    adaptive_ratio=0.3,
                                                    total_epochs=num_epochs,
                                                    nr_epochs_full_teaching=5,
                                                    teacher_help_epoch_percent=teacher_help_percent,
                                                    apply_on_intervals=adaptive_teacher_on_intervals
                                                    )
    else:
        model.setup_adaptive_teacher_forcing(adaptive_teacher=adaptive_teacher_forcing,
                                             adaptive_ratio=0.3,
                                             total_epochs=num_epochs,
                                             nr_epochs_full_teaching=5,
                                             teacher_help_epoch_percent=teacher_help_percent,
                                             apply_on_intervals=adaptive_teacher_on_intervals)

    device = model.module.get_device() if torch.cuda.device_count() > 1 else model.get_device()
    optimizer = create_optimizer(model=model, learning_rate=learning_rate, optimizer_choice=optimizer_choice,
                                 lr_decay=0.7)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=41, gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, min_lr=1e-6,
                                                     threshold=scheduler_threshold)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    if load_model:
        model_path = model_name + ".pth.tar"
        load_checkpoint(torch.load(model_path), model, optimizer)

    # Log model and training parameters on tensorborad
    training_params = {
        "Model Name": model_name,
        "Number of Epochs": num_epochs,
        "Starting Learning Rate": learning_rate,
        "Batch Size": batch_size,
        "Optimizer Choice": optimizer_choice,
        "Save at Number of Steps": save_at_nr_steps,
        "Respect Dataset Max Length": respect_dataset_max_length,
        "Teacher Forcing Ratio": teacher_forcing_ratio
    }
    training_params.update(model_kwargs)
    log_training_parameters(writer, training_params)

    #activez modul de train
    model.train()
    step_nr = 0 # va fi folosit doar ca sa dau print la stepul curent
    #intial am facut greseala sa calculez lossul cu el

    #numarul total de pasi e calculat astfel: numarul de date din train / batch size ales * cate epochi sa repete training loopul
    total_nr_steps = num_epochs * len(tokenized_train["text"]) / batch_size


    train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_tensor_dataset, batch_size=batch_size, shuffle=True)
    min_epoch_loss = float('inf')

    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0
        total_samples_on_epoch = 0
        model.train()
        for step, batch in enumerate(train_loader):
            source = batch[0].to(device)
            target = batch[1].to(device)

            source = source.transpose(0, 1)
            target = target.transpose(0, 1)

            total_samples_on_epoch += source.size(0)

            optimizer.zero_grad()
            # Forward pass
            output = model.forward(source, target, teacher_force_ratio=teacher_forcing_ratio, current_epoch=epoch)

            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            # Backward propagation of loss
            loss = criterion(output, target)
            loss.backward()

            #Mentin in intervalul 0-1 valorile ca sa nu sufere modelul de gradient exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()
            step_loss = loss.item()
            epoch_loss += step_loss
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Step [{step_nr}/{total_nr_steps}] => loss = {step_loss:.3f} || epoch_loss = {epoch_loss / (step + 1):.3f}, Learning Rate {current_lr:.6f}')


            writer.add_scalar("Step Training loss", step_loss, global_step=step_nr)
            step_nr += 1

            if step_nr % save_at_nr_steps == 0 and save_at_nr_epoch is None:
                name = model_name + f'-{epoch_loss/(step + 1):.3f}.pth.tar'
                checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                save_checkpoint(checkpoint, filename=name, path="/out/LSTM_Checkpoints/")
        end_time = time.time()
        epoch_loss /= len(train_loader)
        scheduler.step(epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # if epoch == teacher_help_percent * num_epochs:
        #     new_learning_rate = (learning_rate + current_lr)/2
        #     set_learning_rate(optimizer, new_learning_rate, 1)

        if manual_schedular_decreasing:
            if schedular_decreasing_epochs is not None:
                if epoch in schedular_decreasing_epochs:
                    set_learning_rate(optimizer, current_lr, 0.5)

        writer.add_scalar('Epoch Training Loss', epoch_loss, epoch)
        writer.add_scalar('train/epoch', epoch, epoch)
        writer.add_scalar('train/learning_rate', current_lr, epoch)
        writer.add_scalar('train/loss', epoch_loss, epoch)
        writer.add_scalar("Learning Rate for each Epoch", current_lr, global_step=epoch)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Current Learning Rate: {current_lr:.6f}')

        if epoch_loss < min_epoch_loss:
            min_epoch_loss = epoch_loss

        train_runtime = end_time - start_time
        train_samples_per_second = total_samples_on_epoch / train_runtime

        writer.add_scalar('train/train_runtime', train_runtime, epoch)
        writer.add_scalar('train/train_samples_per_second', train_samples_per_second, epoch)

        ################################# Validation Side ########################################

        val_epoch_loss, val_samples_per_second, val_runtime, accuracy = validation_epoch(model=model,
                                                                               data_loader=validation_loader,
                                                                               criterion=criterion,
                                                                               device=device)
        writer.add_scalar('eval/loss', val_epoch_loss, epoch)
        writer.add_scalar('eval/accuracy', accuracy, epoch)
        writer.add_scalar('eval/runtime', val_runtime, epoch)
        writer.add_scalar('eval/samples_per_second', val_samples_per_second, epoch)
        print(f'Evaluation for Epoch [{epoch+1}/{num_epochs}], Loss: {val_epoch_loss:.4f}, Accuracy: {accuracy:.4f}')

        ################################# End Validation #########################################

        # if epoch+1 == int(0.3 * num_epochs):
        #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        # if epoch+1 == int(0.7 * num_epochs):
        #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        if save_at_nr_epoch is not None:
            if (epoch+1) % save_at_nr_epoch == 0:
                name = model_name + f'-Saving_epoch-{epoch+1}-{epoch_loss:.3f}.pth.tar'
                checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                save_checkpoint(checkpoint, filename=name, path="/out/LSTM_Checkpoints/")

        if epoch+1 == num_epochs:
            name = model_name + f'-final-{epoch_loss:.3f}.pth.tar'
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint, filename=name, path="/out/LSTM_Checkpoints/")
    writer.close()

    return "done"

train(
    model_name=model_name,
    num_epochs=20,
    learning_rate=1e-3,
    batch_size=64,
    load_model=False,
    respect_dataset_max_length=True,
    save_at_nr_steps=10000,
    save_at_nr_epoch=100,
    optimizer_choice="Adam",
    adaptive_teacher_forcing=True,
    teacher_forcing_ratio=0.5,
    teacher_help_percent=1.0,
    adaptive_teacher_on_intervals=[(0, 10), (13, 18)],
    hidden_size=1024,
    num_layers=4,
    encoder_dropout=0.3,
    decoder_dropout=0.3,
    use_lstm=True,
    use_attention=True,
    scheduler_patience=5,
    scheduler_threshold=0.01,
    scheduler_factor=0.1,
    manual_schedular_decreasing=False,
    schedular_decreasing_epochs=[35]
    )

