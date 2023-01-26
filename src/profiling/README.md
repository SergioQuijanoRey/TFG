# Profiling results

> Markdown file where we discuss what we learn from the profiling saves

- To explore a profiling save, run `python -m pstats <profiling_save_path>`
- In a more visual way, run `snakeviz <profiling_save_path>`

# First profiling

> Profiling done with lazy data augmentation, `P = 200, K = 2`

- Structure of the saved files:
    - Binary profile stats: `first_profile.stat`
    - Text version, ordered by `cumtime`: `first_profile.txt`
    - Text filtered version: `first_profile_filtered.txt`
        - Contains only entries of our lib code
- Total time: 328.638 seconds
- Most of the time is spent in `dataloader.__next__`. We don't own that implementation, but probably there is performance issues with the lazy data augmentation, or the way we transform the images
- The second part where most of the time is spent is in `Model.forward` where `Model` is the current model we're training, and `loss_functions.forward`
- Conclusions:
    - First, try to use the not lazy evaluated data augmentation, and second try to optimize the loss functions operations (because models are harder to optimize, as they're mostly implemented by pytorch lib)

# Second profiling

> Profiling done with non-lazy data augmentation, `P = 100, K = 2`. Using smaller batch size (P is now half the size), otherwise we ran out of RAM in Google Colab

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
    - TODO: we could use a statistical hypothesis test for saying that the difference is not relevant, but might a little too much for this

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

# Perf changes log

> We are going to make some changes to the code in order to have better run times. So first we will write benchmarks for the functions that we are going to change. Then we make some changes and keep track of new benchmarks results and also the total run time for training loop.

- About the two benchmarks we're running:
    - Benchmark entries in the following table are (mean, sd)
    - The benchmarks we're running are defined in `./src/benchmarks/{benchmark_loss_functions, benchmark_metrics}.py`
- About the training loop:
    - We use the following parameters:
        - Lazy data augmentation
        - P, K = 100, 2
        - Embedding dimension = 5
        - Training epochs = 1
    - We use the notebook defined in `./src/Benchmarking notebook.ipynb`
    - We only have the total time. No repetitions are made, so we don't have mean, sd information

| `compute_intercluster_metrics` | `precompute_pairwise_distances` | Training loop | Change Description                                        | Git commit                               |
| :---                           | :---                            | :---          | :---                                                      | :---                                     |
| 27.3141, 9.1206                | 39.1880, 0.3676                 | 2341.9525     | No changes made. The project is as we started the process | e7d5342a16d466aff904891168eb6a834561fb2d |
