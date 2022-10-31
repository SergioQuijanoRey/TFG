# Profiling results

> Markdown file where we discuss what we learn from the profiling saves

- To explore a profiling save, run `python -m pstats <profiling_save_path>`
- In a more visual way, run `snakeviz <profiling_save_path>`

# First profiling

> Profiling done with lazy data augmentation

- Stored in file `./profiling_saves/first_profile.stat`
- A text version is created at `./profiling_saves/first_profile.txt`, ordered by `cumtime`
- `./profiling_saves/first_profile_filtered.txt` contains only entries of our lib code
- Total time: 328.638 seconds
- Most of the time is spent in `dataloader.__next__`. We don't own that implementation, but probably there is performance issues with the lazy data augmentation, or the way we transform the images
- The second part where most of the time is spent is in `Model.forward` where `Model` is the current model we're training, and `loss_functions.forward`
- Conclusions:
    - First, try to use the not lazy evaluated data augmentation, and second try to optimize the loss functions operations (because models are harder to optimize, as they're mostly implemented by pytorch lib)
