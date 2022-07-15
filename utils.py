import torch as T

def select(probs, is_training=True):
    '''
    ### Select next to be executed.
    -----
    Parameter:
        probs: probabilities of each operation

    Return: index of operations, log of probabilities
    '''
    if is_training:
        dist = T.distributions.Categorical(probs)
        op = dist.sample()
        logprob = dist.log_prob(op)
    else:
        prob, op = T.max(probs, dim=1)
        logprob = T.log(prob)
    logprob = logprob.sum(dim=1)
    return op, logprob



def save(agent, agent_opt, critic, critic_opt, step, checkpoint_path):
    checkpoint = {}
    checkpoint["agent_state_dict"] = agent.state_dict()
    checkpoint["agent_opt_state_dict"] = agent_opt.state_dict()
    checkpoint["critic_state_dict"] = critic.state_dict()
    checkpoint["critic_opt_state_dict"] = critic_opt.state_dict()
    checkpoint["last_step"] = step
    T.save(checkpoint, checkpoint_path.absolute())
    checkpoint_backup_path = checkpoint_path.parent /(checkpoint_path.name + "_")
    T.save(checkpoint, checkpoint_backup_path.absolute())