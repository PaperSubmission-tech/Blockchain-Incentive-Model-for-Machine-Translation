# Tokenizing using BERT Tokenizer

from transformers import BertTokenizer
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

input_ids = []
attention_masks = []
for sent in sentences:
encoded_dict = tokenizer.encode_plus(sent,
add_special_tokens = True, max_length = 64,
pad_to_max_length = True,return_attention_mask = True,return_tensors = 'pt',
)
input_ids.append(encoded_dict['input_ids'])
attention_masks.append(encoded_dict['attention_mask'])
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import wandb
def ret_dataloader():
batch_size = wandb.config.batch_size
print('batch_size = ', batch_size)
train_dataloader = DataLoader(train_dataset,sampler = RandomSampler(train_datase       t),  batch_size = batch_size)


validation_dataloader = DataLoader( val_dataset,  sampler = SequentialSampler(val_     dataset),   batch_size = batch_size)
return train_dataloader,validation_dataloader

# BERT Model

from transformers import BertForSequenceClassification, AdamW, BertConfig
def ret_model():
model = BertForSequenceClassification.from_pretrained(
"bert-base-uncased",
num_labels = 2,
output_attentions = False, # Whether the model returns attentions weights.
output_hidden_states = False, # Whether the model returns all hidden-states.
)
return model

# Optimizer
def ret_optim(model):
print('Learning_rate = ',wandb.config.learning_rate )
optimizer = AdamW(model.parameters(),
lr = wandb.config.learning_rate,
eps = 1e-8
)
return optimizer


from transformers import get_linear_schedule_with_warmup
def ret_scheduler(train_dataloader,optimizer):
epochs = wandb.config.epochs
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0,           num_training_steps = total_steps)
return scheduler

# Training
for epoch_i in range(0, epochs):
        

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()        
            loss, logits = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)
            #Log the metric
            wandb.log({'train_batch_loss':loss.item()})

            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()