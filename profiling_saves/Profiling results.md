# Profiling results

> Markdown file where we discuss what we learn from the profiling saves

- To explore a profiling save, run `python -m pstats <profiling_save_path>`
- In a more visual way, run `snakeviz <profiling_save_path>`

# First profiling

> Profiling done with lazy data augmentation, `P = 200, K = 2`

- Structure of the saved files:
    - Binary profile stats: `./profiling_saves/first_profile.stat`
    - Text version, ordered by `cumtime`: `./profiling_saves/first_profile.txt`
    - Text filtered version: `./profiling_saves/first_profile_filtered.txt`
        - Contains only entries of our lib code
- Total time: 328.638 seconds
- Most of the time is spent in `dataloader.__next__`. We don't own that implementation, but probably there is performance issues with the lazy data augmentation, or the way we transform the images
- The second part where most of the time is spent is in `Model.forward` where `Model` is the current model we're training, and `loss_functions.forward`
- Conclusions:
    - First, try to use the not lazy evaluated data augmentation, and second try to optimize the loss functions operations (because models are harder to optimize, as they're mostly implemented by pytorch lib)

# Second profiling

> Profiling done with non-lazy data augmentation, `P = 100, K = 2`. Using smaller batch size, otherwise we ran out of RAM in Google Colab

- Same structure as first profiling, but with name `second_...`
- This time, it tooks 947.719 second to train. Its much more, but it can be because of different P-K values
- Using `snakeviz`, most of the time is spent in `train_loggers`. In the loggers:
    - Most of the time is spent in `compute_intercluster_metrics`. Inside that function, most of the time is spent in `metrics.py:__compute_pairwise_distances`
- Loss function is used a lot, in different places. So optimizing this set of functions might be worth (even though there is a lot of pure pytorch tensor code)

# Experiment

- We run the training with `P = 100, K = 2` with lazy and non-lazy data augmentation, and compare training times
- Note that we don't profile the training, we're only interested in training times
- Non-lazy: 917.8408761024475 seconds
- Lazy: 952.0073399543762 seconds
- **Conclusion**: there is not a big difference (15.29 mins vs 15.86 mins)

# Other conclusions

- When using non-lazy data loading, GPU usage a little bit higher

# Pros and cons

| Method   | Pro or Con | Description                                                                                                                                                       |
| :---     | :---       | :---                                                                                                                                                              |
| Lazy     | Pro        | P-K values can be as big as we want, laziness makes low use of RAM                                                                                                |
| Lazy     | Pro        | Every time we access to an augmented item, its created at runtime. So in different epochs we have different images. Thus, more augmentation than with lazy method |
| Lazy     | Pro        | Is not much slower than the non-lazy, thus might seem like the preferred option given the advantages it has                                                       |
| Non-lazy | Con        | P-K values are limited, because big values can cause that we ran out of RAM                                                                                       |
| Non-lazy | Con        | In principle, it should be faster, but its not significantly faster                                                                                               |
| Non-lazy | Pro        | Data augmentation caching can be extended to cache all P-K values, and not just the last P-K value                                                                |
