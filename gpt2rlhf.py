import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import Env, spaces
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')

# Step 1: Load the GPT-2 Model and Tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)

# Add padding token to tokenizer
tokenizer.pad_token = tokenizer.eos_token
gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id

# Step 2: Generate Baseline Text Outputs
def generate_text(model, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        inputs.input_ids, 
        max_length=max_length, 
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Baseline evaluation
prompts = [
    "The future of AI is",
    "In a world where",
    "Climate change will",
    "The key to happiness is",
    "Technology's impact on society"
]
baseline_results = [generate_text(gpt2_model, prompt) for prompt in prompts]

print("Baseline Results:")
for prompt, result in zip(prompts, baseline_results):
    print(f"Prompt: {prompt}")
    print(f"Result: {result}\n")

# Step 3: Define Human Feedback (Simulated)
def simulate_feedback(text):
    # More sophisticated feedback simulation
    words = word_tokenize(text.lower())
    unique_words = len(set(words))
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    
    diversity_score = min(unique_words / len(words), 1) * 2  # 0-2 points
    length_score = min(len(words) / 30, 1) * 1.5  # 0-1.5 points
    complexity_score = min(sentence_count / 3, 1) * 1.5  # 0-1.5 points
    
    return diversity_score + length_score + complexity_score

# Step 4: Implement RLHF
class CustomEnv(Env):
    def __init__(self, gpt2_model, tokenizer, prompt):
        super(CustomEnv, self).__init__()
        self.gpt2_model = gpt2_model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_length = 50
        
        self.action_space = spaces.Discrete(tokenizer.vocab_size)
        self.observation_space = spaces.Box(low=0, high=tokenizer.vocab_size, shape=(self.max_length,), dtype=np.int32)
        
        self.current_text = ""
        self.current_length = 0

    def step(self, action):
        token = self.tokenizer.decode([action])
        self.current_text += token
        self.current_length += 1
        
        done = self.current_length >= self.max_length or token == self.tokenizer.eos_token
        reward = simulate_feedback(self.current_text) if done else 0
        
        obs = self.get_observation()
        return obs, reward, done, False, {}

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_text = self.prompt
        self.current_length = len(self.tokenizer.encode(self.prompt))
        return self.get_observation(), {}

    def get_observation(self):
        encoded = self.tokenizer.encode(self.current_text, padding='max_length', max_length=self.max_length, truncation=True)
        return np.array(encoded)

# Initialize environment
def make_env(prompt):
    return lambda: CustomEnv(gpt2_model, tokenizer, prompt)

env = DummyVecEnv([make_env(prompts[0])])

# Train with PPO
ppo_model = PPO("MlpPolicy", env, verbose=1, n_steps=1024, batch_size=64, n_epochs=10)
ppo_model.learn(total_timesteps=10000)

# Step 5: Evaluate Post-Training Performance
def evaluate_model(model, prompts, num_samples=5):
    results = []
    for prompt in prompts:
        samples = [generate_text(model, prompt) for _ in range(num_samples)]
        bleu_scores = [sentence_bleu([word_tokenize(ref)], word_tokenize(hyp)) for ref in samples for hyp in samples if ref != hyp]
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        avg_feedback = sum(simulate_feedback(sample) for sample in samples) / num_samples
        results.append((samples, avg_bleu, avg_feedback))
    return results

print("Post-Training Results:")
post_training_results = evaluate_model(gpt2_model, prompts)
for prompt, (samples, avg_bleu, avg_feedback) in zip(prompts, post_training_results):
    print(f"Prompt: {prompt}")
    print(f"Samples:")
    for sample in samples:
        print(f"  - {sample}")
    print(f"Average BLEU score: {avg_bleu:.4f}")
    print(f"Average Feedback score: {avg_feedback:.4f}\n")

# Compare baseline and post-training performance
print("Performance Comparison:")
baseline_scores = [simulate_feedback(result) for result in baseline_results]
post_training_scores = [avg_feedback for _, _, avg_feedback in post_training_results]

print(f"Average Baseline Feedback: {sum(baseline_scores) / len(baseline_scores):.4f}")
print(f"Average Post-Training Feedback: {sum(post_training_scores) / len(post_training_scores):.4f}")
