import torch
import tiktoken
from train_model import GPT, GPT_config

# Load the model
torch.set_float32_matmul_precision('high')
model = GPT(GPT_config)
model.to('cuda')
model = torch.compile(model)
model.load_state_dict(torch.load('models/gpt2_kenspeed.pth'))
model.eval()

device = 'cuda'
enc = tiktoken.get_encoding('gpt2')
user = ''
while user != 'q':
    user_range = int(input('Enter a length (q to quit): '))
    temp = float(input('Enter a temperature (q to quit): '))
    user = input('Enter a prompt (q to quit): \n')
    print('--------------------------------------------------')
    print(user, end='', flush=True)
    tokens = enc.encode(user)
    tokens = torch.tensor(tokens, dtype=torch.long, device='cuda')
    tokens = tokens.view((1, -1))
    with torch.no_grad():
        for i in range(user_range):
            # with torch.autocast(device_type=device, dtype=torch.bfloat16):
            #     logits, loss = model(tokens)
            logits, loss = model(tokens)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits / temp, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
            print(enc.decode([token.item()]),end='',flush=True)  # type: ignore
            tokens = torch.cat((tokens, token), dim=-1)
        print('\n--------------------------------------------------', flush=True)
