import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import Env, spaces
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
import ssl
from textstat import flesch_reading_ease, syllable_count
from rouge import Rouge
from stable_baselines3.common.policies import ActorCriticPolicy
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Step 1: Load GPT-2 Model and Tokenizer
model_name = "gpt2"  # You can choose 'gpt2-medium' or larger if available
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)

# Add padding token to tokenizer
tokenizer.pad_token = tokenizer.eos_token
gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id

# Step 2: Generate Baseline Text Outputs
def generate_text(model, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.92,  # Changed from top_k=50, top_p=0.95
        temperature=0.7  # Increased from 0.1
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

# Step 3: Define Enhanced Human Feedback Function
def simulate_feedback(text, prompt):
    # Tokenize text and prompt
    words = word_tokenize(text.lower())
    prompt_words = word_tokenize(prompt.lower())
    unique_words = len(set(words))
    total_words = len(words)
    sentence_count = text.count('.') + text.count('!') + text.count('?')

    # Diversity score
    diversity_score = unique_words / total_words if total_words > 0 else 0  # Normalize to 0-1

    # Length score
    desired_length = 100  # Adjust as needed
    length_score = min(total_words / desired_length, 1)  # Normalize to 0-1

    # Coherence score (simple overlap for illustration)
    common_words = len(set(words) & set(prompt_words))
    coherence_score = common_words / len(prompt_words) if len(prompt_words) > 0 else 0  # Normalize to 0-1

    # Readability score
    readability = flesch_reading_ease(text)
    readability_score = (readability + 206.835) / 412.165  # Normalize to 0-1

    # Penalty for repetition
    repetition_penalty = (total_words - unique_words) / total_words if total_words > 0 else 0  # 0-1

    # Aggregate scores
    feedback = (
        0.25 * diversity_score +
        0.25 * length_score +
        0.25 * coherence_score +
        0.25 * readability_score
    )
    # Subtract repetition penalty
    feedback *= (1 - repetition_penalty)

    # Ensure feedback is within 0-10
    feedback = max(min(feedback * 10, 10), 0)
    return feedback

# Step 4: Implement RLHF with Improved Environment
class CustomEnv(Env):
    def __init__(self, gpt2_model, tokenizer, prompts):
        super(CustomEnv, self).__init__()
        self.gpt2_model = gpt2_model
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.max_length = 100
        self.action_space = spaces.Discrete(self.tokenizer.vocab_size)

        self.observation_space = spaces.Box(
            low=0,
            high=self.tokenizer.vocab_size - 1,
            shape=(self.max_length,),
            dtype=np.int32
        )

        self.current_prompt = ""
        self.current_text = ""
        self.current_length = 0  # Initialize current length

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_prompt = np.random.choice(self.prompts)
        self.current_text = self.current_prompt
        self.current_length = len(self.tokenizer.encode(self.current_prompt))
        return self.get_observation(), {}  # Return observation and empty info dict

    def step(self, action):
        token = self.tokenizer.decode([action])
        self.current_text += token
        self.current_length += 1

        done = self.current_length >= self.max_length or action == self.tokenizer.eos_token_id

        # Provide intermediate reward every 10 tokens or at the end
        reward = 0
        if self.current_length % 10 == 0 or done:
            reward = simulate_feedback(self.current_text, self.current_prompt) / 10.0  # Scale reward

        if done and self.current_length < self.max_length * 0.5:
            # Penalty for early stopping
            reward -= 1.0  # Adjust penalty as needed

        obs = self.get_observation()
        return obs, reward, done, False, {}  # Return obs, reward, done, truncated, info

    def get_observation(self):
        encoded = self.tokenizer.encode(
            self.current_text,
            truncation=True,
            max_length=self.max_length
        )
        if len(encoded) < self.max_length:
            encoded += [self.tokenizer.pad_token_id] * (self.max_length - len(encoded))
        return np.array(encoded)

# Initialize environment with multiple prompts
def make_env():
    return CustomEnv(gpt2_model, tokenizer, prompts)

env = DummyVecEnv([make_env])

# Step 5: Define Custom LSTM Policy Network
from stable_baselines3.common.distributions import CategoricalDistribution

class CustomLSTMPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(CustomLSTMPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )

        # Define embedding layer
        self.embedding_layer = nn.Embedding(
            num_embeddings=action_space.n,
            embedding_dim=128
        )

        # Define LSTM layer
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            batch_first=True
        )

        # Shared linear layer
        self.shared_net = nn.Linear(64, 64)

        # Action and value heads
        self.action_net = nn.Linear(64, action_space.n)
        self.value_net = nn.Linear(64, 1)

        # Initialize weights
        self._init_weights()

        # Action distribution
        self.action_dist = CategoricalDistribution(action_space.n)

    def _init_weights(self):
        # Initialize weights as per stable_baselines3 recommendations
        modules = [self.embedding_layer, self.lstm, self.shared_net, self.action_net, self.value_net]
        for module in modules:
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

    def extract_features(self, obs):
        # Convert obs to LongTensor
        if not isinstance(obs, torch.LongTensor):
            obs = obs.long().to(self.device)

        embedded = self.embedding_layer(obs)  # [batch_size, seq_length, embedding_dim]
        lstm_out, _ = self.lstm(embedded)
        features = torch.relu(self.shared_net(lstm_out[:, -1, :]))
        return features

    def forward_actor(self, obs):
        features = self.extract_features(obs)
        return self.action_net(features)

    def forward_critic(self, obs):
        features = self.extract_features(obs)
        return self.value_net(features)

    def _get_action_dist_from_logits(self, action_logits):
        return self.action_dist.proba_distribution(action_logits)  # Corrected line

    def forward(self, obs, deterministic=False):
        action_logits = self.forward_actor(obs)
        values = self.forward_critic(obs).squeeze(-1)

        distribution = self._get_action_dist_from_logits(action_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def get_distribution(self, obs):
        action_logits = self.forward_actor(obs)
        return self._get_action_dist_from_logits(action_logits)

    def evaluate_actions(self, obs, actions):
        action_logits = self.forward_actor(obs)
        distribution = self._get_action_dist_from_logits(action_logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.forward_critic(obs).squeeze(-1)
        return values, log_prob, entropy

    def predict(self, obs, deterministic=False):
        actions, _, _ = self.forward(obs, deterministic=deterministic)
        return actions, None

    def get_action_logits(self, obs):
        action_logits = self.forward_actor(obs)
        return action_logits

# Step 6: Train with PPO (Improved PPO Settings)
ppo_model = PPO(
    policy=CustomLSTMPolicy,
    env=env,
    verbose=1,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    learning_rate=1e-4,  # Increased learning rate
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    max_grad_norm=0.5,
    tensorboard_log="./ppo_tensorboard/"
)

# Move GPT-2 model to the same device as the PPO model
device = ppo_model.policy.device
gpt2_model.to(device)
tokenizer.model_max_length = 100  # Ensure tokenizer handles the max length

ppo_model.learn(total_timesteps=200000)

# Step 7: LMS-like Adaptive Scaling Function
scaling_factor = 1.0
learning_rate = 0.005  # Adjusted learning rate

def lms_update(feedback_score, target_score=7.0):
    global scaling_factor, learning_rate
    error = target_score - feedback_score
    scaling_factor += learning_rate * error
    # Clip scaling factor to prevent instability
    scaling_factor = max(min(scaling_factor, 5.0), 0.1)
    # Decay learning rate
    learning_rate *= 0.999

# Step 8: Generate Text with PPO and LMS
def generate_text_with_ppo_lms(gpt2_model, ppo_model, prompt, max_length=100):
    device = ppo_model.policy.device
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids.clone()

    for _ in range(max_length):
        # Get GPT-2 output
        outputs = gpt2_model(generated)
        next_token_logits = outputs.logits[:, -1, :]

        # Get PPO policy output
        obs = generated[:, -env.observation_space.shape[0]:]  # Adjust based on observation space
        obs = obs.to(device)

        # Get action logits from the policy
        with torch.no_grad():
            action_logits = ppo_model.policy.get_action_logits(obs)

        # Combine logits
        combined_logits = next_token_logits + scaling_factor * action_logits
        combined_probs = torch.softmax(combined_logits, dim=-1)

        # Sample next token
        next_token = torch.multinomial(combined_probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=-1)

        # Check for EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    feedback_score = simulate_feedback(generated_text, prompt)
    lms_update(feedback_score)  # Update scaling factor using LMS-like method

    return generated_text

# Step 9: Evaluate Post-Training Performance with ROUGE and Feedback
def evaluate_model(ppo_model, prompts, num_samples=5):
    rouge = Rouge()
    results = []
    for prompt in prompts:
        samples = [generate_text_with_ppo_lms(gpt2_model, ppo_model, prompt) for _ in range(num_samples)]
        rouge_scores = []
        for i in range(len(samples)):
            for j in range(i+1, len(samples)):
                scores = rouge.get_scores(samples[i], samples[j])[0]
                rouge_scores.append(scores['rouge-l']['f'])
        avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
        avg_feedback = sum(simulate_feedback(sample, prompt) for sample in samples) / num_samples
        results.append((samples, avg_rouge, avg_feedback))
    return results

print("Post-Training Results:")
post_training_results = evaluate_model(ppo_model, prompts)
for prompt, (samples, avg_rouge, avg_feedback) in zip(prompts, post_training_results):
    print(f"Prompt: {prompt}")
    print(f"Samples:")
    for sample in samples:
        print(f"  - {sample}")
    print(f"Average ROUGE-L F1 score: {avg_rouge:.4f}")
    print(f"Average Feedback score: {avg_feedback:.4f}\n")

# Compare baseline and post-training performance
print("Performance Comparison:")
baseline_scores = [simulate_feedback(result, prompt) for result, prompt in zip(baseline_results, prompts)]
post_training_scores = [avg_feedback for _, _, avg_feedback in post_training_results]

print(f"Average Baseline Feedback: {sum(baseline_scores) / len(baseline_scores):.4f}")
print(f"Average Post-Training Feedback: {sum(post_training_scores) / len(post_training_scores):.4f}")

# Step 10: Test the LMS-guided Text Generation
print("\nLMS-Guided Text Generation:")
ppo_scores = []
for prompt in prompts:
    lms_generated_text = generate_text_with_ppo_lms(gpt2_model, ppo_model, prompt)
    feedback = simulate_feedback(lms_generated_text, prompt)
    ppo_scores.append(feedback)
    print(f"Prompt: {prompt}")
    print(f"Generated Text: {lms_generated_text}")
    print(f"Feedback Score: {feedback:.4f}\n")

# Final Performance Comparison
print("Final Performance Comparison:")
print(f"Average Baseline Feedback: {sum(baseline_scores) / len(baseline_scores):.4f}")
print(f"Average Post-Training Feedback: {sum(post_training_scores) / len(post_training_scores):.4f}")
print(f"Average PPO-Guided Feedback: {sum(ppo_scores) / len(ppo_scores):.4f}")
