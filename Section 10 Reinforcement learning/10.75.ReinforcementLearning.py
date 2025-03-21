##########################
# Reinforcement Learning #
# Pseudo Implementation  #
##########################

# the objects are taken from previous code

# Step 1
# these are the variables we need to init before we run the model
obs = env.reset()
# obs: the first real input frame from the environment
# This resets the environment and gets the initial observation

h = mdnrnn.initial_state
# h: the initial state of the MDN-RNN
# This sets the initial hidden state of the MDN-RNN

done = False
# done: a variable to indicate if the episode is finished
# This is set to False since the game has just started

cumulative_reward = 0
# cumulative_reward: a variable to keep track of the total reward
# This starts at 0 each time a new episode begins

# Step 2
# loop for the game
while (not done):
    # get latent vector z which is returned by the VAE
    z = cnnvae(obs)
    # z: the latent vector returned by the VAE
    # We input the observation into the VAE, and this returns the latent vector z

    # Now Z is pushed to the controller, and the MDN-RNN
    a = controller([z, h])
    # a: the action returned by the controller
    # The controller takes the concatenated input vector (z) and the hidden state (h) of the MDN-RNN

    # Now we will call the environment
    # we will get the next state, reward, done
    obs, reward, done = env.step(a)
    # obs: the next state returned by the environment
    # reward: the reward returned by the environment
    # done: a variable to indicate if the episode is finished
    # We input the action into the environment, and this returns the next state, reward, and done

    # Since we have the reward now, we will increment the cumulative reward
    cumulative_reward += reward
    # This adds the reward to the cumulative reward

    # Now we will get the next hidden state
    h = mdnrnn([a, z, h])
    # h: the next hidden state of the MDN-RNN
    # The MDN-RNN takes the latent vector (z), action (a), and the hidden state (h) of the MDN-RNN to get the next hidden state
