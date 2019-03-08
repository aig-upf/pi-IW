import numpy as np

def sample_cdf(cum_probs, size=None):
    s = cum_probs[-1]
    assert s > 0.99999 and s < 1.00001, "Probabilities do not sum to 1: %"%cum_probs #just to check our input looks like a probability distribution, not 100% sure though.
    if size is None:
        # if rand >=s, cumprobs > rand would evaluate to all False. In that case, argmax would take the first element argmax([False, False, False]) -> 0.
        # This may happen still if probabilities sum to 1:
        # cumsum > rand is computed in a vectorized way, and in our machine (looks like) these operations are done in 32 bits.
        # Thus, even if our probabilities sum to exactly 1.0 (e.g. [0. 0.00107508 0.00107508 0.0010773 0.2831216 1.]), when rand is really close to 1 (e.g. 0.999999972117424),
        # when computing cumsum > rand in a vectorized way it will consider it in float32, which turns out to be cumsum > 1.0 -> all False.
        # This is why we check that (float32)rand < s:
        while True:
            rand = np.float32(np.random.rand())
            if rand < s:
                break
        res = (cum_probs > rand)
        return res.argmax()

    if type(size) is int:
        rand = np.random.rand(size).reshape((size,1))
    else:
        assert type(size) in (tuple, list), "Size can either be None for scalars, an int for vectors or a tuple/list containing the size for each dimension."
        assert len(size) > 0, "Use None for scalars."
        rand = np.random.rand(*size).reshape(size+(1,))
    # Again, we check that (float32)rand < s (easier to implement)
    mask = rand.astype(np.float32) >= s
    n = len(rand[mask])
    while n > 0:
        rand[mask] = np.random.rand(n)
        mask = rand.astype(np.float32) >= s
        n = len(rand[mask])
    return (cum_probs > rand).argmax(axis=-1)


def sample_pmf(probs, size=None):
    return sample_cdf(probs.cumsum(), size)


def random_index(array_len, size=None, replace=False, probs=None, cumprobs=None):
    """
    Similar to np.random.choice, but slightly faster.
    """
    if probs is None and cumprobs is None:
        res = np.random.randint(0, array_len, size)
        one_sample = lambda: np.random.randint(0, array_len)
    else:
        assert probs is None or cumprobs is None, "Either both probs and cumprobs is None (uniform probability distribution used) or only one of them is not None, not both."
        if cumprobs is None:
            cumprobs = probs.cumsum()
        assert array_len == len(cumprobs)
        res = sample_cdf(cumprobs, size)
        one_sample = lambda: sample_cdf(cumprobs)

    if not replace and size is not None:
        assert size <= array_len, "The array has to be longer than 'size' when sampling without replacement."
        s = set()
        for i in range(size):
            l = len(s)
            s.add(res[i])
            while len(s) == l:
                res[i] = one_sample()
                s.add(res[i])
    return res


def softmax(x, temp=1, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    if temp == 0:
        res = (x == np.max(x, axis=-1))
        return res/np.sum(res, axis=-1)
    x = x/temp
    e_x = np.exp( (x - np.max(x, axis=axis, keepdims=True)) ) #subtracting the max makes it more numerically stable, see http://cs231n.github.io/linear-classify/#softmax and https://stackoverflow.com/a/38250088/4121803
    return e_x / e_x.sum(axis=axis, keepdims=True)