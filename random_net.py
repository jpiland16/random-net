import random
from tqdm import tqdm

print("\nLoading PyTorch...\n")

import torch
from torch import Tensor, nn

def generate_single_vector_set(num_policies: int, time_range: int, 
        reward_mag: int, use_positive_reward: bool, 
        use_time: bool) -> 'tuple[list[int], list[int]]':
    """
    Generates a sparse vector of the form [0, 1, 0, ..., 0, X]
    where X is less than time_range and the vector is of size 
    num_policies + 1.

    Also returns the target vector, which looks like
    [0, REWARD_MAG, ..., 0] if POSITIVE_REWARD == True else
    [-REWARD_MAG, 0, ..., -REWARD_MAG]
    """
    vector = [0] * num_policies
    selected_policy = random.randint(0, num_policies - 1)
    vector[selected_policy] = 1
    if use_time:
        vector.append(random.uniform(0, time_range - 1))

    target = [0 if use_positive_reward else -reward_mag] * num_policies
    target[selected_policy] = reward_mag if use_positive_reward else 0

    return vector, target

def generate_multiple_vector_sets(num_policies: int, time_range: int, 
        reward_mag: int, use_positive_reward: bool, use_time: bool,
        vector_count: int) -> 'tuple[list[list[int]], list[list[int]]]':
   
    inputs = []
    targets = []

    for _ in range(vector_count):
        single_in, single_tgt = generate_single_vector_set(num_policies, 
            time_range, reward_mag, use_positive_reward, use_time)
        inputs.append(single_in)
        targets.append(single_tgt)

    return inputs, targets


class LinearNet(nn.Module):

    def __init__(self, learn_rate: float, num_policies: int, use_sigmoid: bool, 
            use_hidden: bool, use_time: bool, use_sigmoid_2: bool):

        super().__init__()
        
        self.input_size = num_policies + (1 if use_time else 0)
        self.output_size = num_policies
        self.use_sigmoid = use_sigmoid
        self.use_hidden = use_hidden
        self.use_sigmoid_2 = use_sigmoid_2

        self.fc = nn.Linear(self.input_size, self.output_size)
        self.fc2 = nn.Linear(self.output_size, self.output_size)
        self.optimizer = torch.optim.Adam(self.parameters(), learn_rate)

    def forward(self, input):

        output = self.fc(input)

        if self.use_sigmoid:
            sigmoid = nn.Sigmoid()
            output = sigmoid(output)

        if self.use_hidden:
            output = self.fc2(output)

        if self.use_sigmoid_2:
            sigmoid_2 = nn.Sigmoid()
            output = sigmoid_2(output)

        return output

    def train(self, input, target):
        output = self(input)
        criterion = nn.MSELoss()
        loss = criterion(output, target)
        loss.backward()
        self.optimizer.step()

        return loss

    def check_accuracy(self, input, target):
        """
        Return the fraction of the time in which the NN guessed the correct
        next policy (i.e. decided to STAY).
        """
        
        output = self(input)
        output_pol_indices = torch.max(output, 1, keepdims=True).indices
        target_pol_indices = torch.max(target, 1, keepdims=True).indices
        
        count = len(input)
        correct = torch.sum(output_pol_indices == target_pol_indices)

        return correct / count

def train_and_test(num_policies: int, num_epochs: int, game_length: int, 
        batch_size: int, reward_mag: int, use_positive_reward: bool, 
        use_time: bool, learn_rate: float, use_sigmoid: bool, 
        use_hidden: bool, use_sigmoid_2: bool = False):

    early_end = False

    net = LinearNet(
        learn_rate = learn_rate, 
        num_policies = num_policies,
        use_sigmoid = use_sigmoid,
        use_hidden = use_hidden,
        use_time = use_time,
        use_sigmoid_2 = use_sigmoid_2
    )

    print(" loss / accuracy")

    iter = tqdm(range(num_epochs))
    for _ in iter:

        ### TRAIN ###

        input, target = generate_multiple_vector_sets(
            num_policies = num_policies, 
            time_range = game_length,
            reward_mag = reward_mag,
            use_positive_reward = use_positive_reward,
            use_time = use_time,
            vector_count = batch_size
        )

        loss = net.train(Tensor(input), Tensor(target)).item()

        ### TEST ###

        input, target = generate_multiple_vector_sets(
            num_policies = num_policies,
            time_range = game_length,
            reward_mag = reward_mag,
            use_positive_reward = use_positive_reward,
            use_time = use_time,
            vector_count = 1000
        )

        accuracy = net.check_accuracy(Tensor(input), Tensor(target)).item()
        
        iter.set_description(f"{loss:5.2f} / {(accuracy * 100):5.2f}%")

        if accuracy == 1:
            early_end = True
            break
    
    if early_end:
        print("Ended early because reached 100% accuracy.")

    print()
