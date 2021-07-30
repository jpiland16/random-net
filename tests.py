from random_net import train_and_test

def main():
    
    print("""
    This one will reach 100% accuracy very quickly, since time is not included.
    Note that game length is irrelevant since we aren't using the time.
    """)
    train_and_test(
        num_policies = 5,
        num_epochs = 2000,
        game_length = 100,
        batch_size = 50,
        reward_mag = 10,
        use_positive_reward = True,
        use_time = False, # notice
        learn_rate = 0.02,
        use_sigmoid = False,
        use_hidden = False
    )

    print("""
    Same as above, except we use sigmoid. 
    Similar results.
    """)
    train_and_test(
        num_policies = 5,
        num_epochs = 2000,
        game_length = 100,
        batch_size = 50,
        reward_mag = 10,
        use_positive_reward = True,
        use_time = False,
        learn_rate = 0.02,
        use_sigmoid = True, # notice
        use_hidden = False
    )

    print("""
    A note, I probably should have changed reward_mag to 1, since can't reach 
    10 as an output while using sigmoid.
    """)
    train_and_test(
        num_policies = 5,
        num_epochs = 2000,
        game_length = 100,
        batch_size = 50,
        reward_mag = 1, # notice
        use_positive_reward = True,
        use_time = False,
        learn_rate = 0.02,
        use_sigmoid = True, # notice
        use_hidden = False
    )

    print("""
    Now add a hidden layer, and take out the sigmoid.
    Adding the layer usually makes it take a bit longer to train.
    Have to keep reward_mag high, else bad things happen.
    """)
    train_and_test(
        num_policies = 5,
        num_epochs = 2000,
        game_length = 100,
        batch_size = 50,
        reward_mag = 10, # notice
        use_positive_reward = True,
        use_time = False,
        learn_rate = 0.02,
        use_sigmoid = False, # notice
        use_hidden = True, # notice
        use_sigmoid_2 = False, # notice
    )

    print("""
    Add intermediate sigmoid but not the final one. Can keep reward 10.
    Sometimes doesn't make it to 100%!
    """)
    train_and_test(
        num_policies = 5,
        num_epochs = 2000,
        game_length = 100,
        batch_size = 50,
        reward_mag = 10, # notice
        use_positive_reward = True,
        use_time = False,
        learn_rate = 0.02,
        use_sigmoid = True, # notice
        use_hidden = True, # notice
        use_sigmoid_2 = False, # notice
    )

    print("""
    Same as above, faster learning rate though.
    """)
    train_and_test(
        num_policies = 5,
        num_epochs = 2000,
        game_length = 100,
        batch_size = 50,
        reward_mag = 10,
        use_positive_reward = True,
        use_time = False,
        learn_rate = 0.05, # notice
        use_sigmoid = True,
        use_hidden = True,
        use_sigmoid_2 = False, 
    )

    print("""
    Add both sigmoids, have to reduce reward mag.
    Sometimes doesn't make it to 100%!
    """)
    train_and_test(
        num_policies = 5,
        num_epochs = 2000,
        game_length = 100,
        batch_size = 50,
        reward_mag = 1, # notice
        use_positive_reward = True,
        use_time = False,
        learn_rate = 0.02,
        use_sigmoid = True, # notice
        use_hidden = True, # notice
        use_sigmoid_2 = True, # notice
    )

    print("""
    Same as above, faster learning rate though.
    """)
    train_and_test(
        num_policies = 5,
        num_epochs = 2000,
        game_length = 100,
        batch_size = 50,
        reward_mag = 1,
        use_positive_reward = True,
        use_time = False,
        learn_rate = 0.05, # notice
        use_sigmoid = True, 
        use_hidden = True, 
        use_sigmoid_2 = True, 
    )

    print("""
    Slower learning rate?
    """)
    train_and_test(
        num_policies = 5,
        num_epochs = 2000,
        game_length = 100,
        batch_size = 50,
        reward_mag = 1,
        use_positive_reward = True,
        use_time = False,
        learn_rate = 0.005, # notice
        use_sigmoid = True, 
        use_hidden = True, 
        use_sigmoid_2 = True, 
    )

    ############## Here we introduce the time variable, which generally  
    ############## is going to mess things up.
    
    print("""
    Introduce time!! Otherwise same as first trial.
    """)
    train_and_test(
        num_policies = 5,
        num_epochs = 2000,
        game_length = 100,
        batch_size = 50,
        reward_mag = 10,
        use_positive_reward = True,
        use_time = True, # notice
        learn_rate = 0.02,
        use_sigmoid = False,
        use_hidden = False
    )

    print("""
    Keep time, but require it to be small.
    """)
    train_and_test(
        num_policies = 5,
        num_epochs = 2000,
        game_length = 1, # notice
        batch_size = 50,
        reward_mag = 10,
        use_positive_reward = True,
        use_time = True, # notice
        learn_rate = 0.02,
        use_sigmoid = False,
        use_hidden = False
    )

    print("""
    Unsuccessfully try big time with small learning rate.
    """)
    train_and_test(
        num_policies = 5,
        num_epochs = 2000,
        game_length = 100,
        batch_size = 50,
        reward_mag = 10,
        use_positive_reward = True,
        use_time = True, # notice
        learn_rate = 0.005, # notice
        use_sigmoid = False,
        use_hidden = False
    )

    print("""
    What about a tiny learning rate?
    """)
    train_and_test(
        num_policies = 5,
        num_epochs = 2000,
        game_length = 100,
        batch_size = 50,
        reward_mag = 10,
        use_positive_reward = True,
        use_time = True, # notice
        learn_rate = 0.0005, # notice
        use_sigmoid = False,
        use_hidden = False
    )

    print("""
    Or a tiny learning rate and bigger batch?
    """)
    train_and_test(
        num_policies = 5,
        num_epochs = 2000,
        game_length = 100,
        batch_size = 500, # notice
        reward_mag = 10,
        use_positive_reward = True,
        use_time = True, # notice
        learn_rate = 0.0005, # notice
        use_sigmoid = False,
        use_hidden = False
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    print()