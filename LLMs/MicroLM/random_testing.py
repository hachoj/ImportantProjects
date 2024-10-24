# Model Testing
from model import SLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.cuda.empty_cache()

checkpoint = torch.load("logs/model_SLM-0.124B_final_control_model_19072.pt", weights_only=False)

config = checkpoint["config"]
from config import GPT_config
config = GPT_config
model = SLM(config)
model.load_state_dict(checkpoint["model"])
step = checkpoint["step"]

teacher = AutoModelForCausalLM.from_pretrained("allenai/OLMo-7B-hf")
tok = AutoTokenizer.from_pretrained("allenai/OLMo-7B-hf")

device = 'cuda'

input = "The nucleus of an oxygen atom contains"
T = 1

x = tok.encode(input, return_tensors='pt').to(device)
x_o = x

model.to(device)
teacher.to(device)

model.eval()
print(input, end='')
for i in range(100):
    m_logits, _ = model(x)
    # sample from the logits that are already in softmax
    probs = torch.nn.functional.softmax(m_logits[0, -1,:]/T, dim=-1)
    sample = torch.multinomial(probs, num_samples=1)
    print(tok.decode(sample), end='')
    sample = sample.view(1,1)
    x = torch.cat((x, sample), dim=1)
print()
print()
print()
print()
print(input, end='')
teacher.eval()
for i in range(100):
    o_logits = teacher(x_o, use_cache=False).logits[:, :, :50280]
    # sample from the logits that are already in softmax
    probs = torch.nn.functional.softmax(o_logits[0, -1,:]/T, dim=-1)
    sample = torch.multinomial(probs, num_samples=1)
    print(tok.decode(sample), end='')
    sample = sample.view(1,1)
    x_o = torch.cat((x_o, sample), dim=1)


print()
print()
print()
print()
print()
print(f"diff: {torch.mean(model(x)[0]-teacher(x, use_cache=False).logits[:, :, :50280])}")
