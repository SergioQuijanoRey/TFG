"""
Module to implement custom loss functions
"""

def triplet_loss_batch_hard(batch):
    """
    Computes batch hard loss. Given batches get in the form of dict [class] => (image, embedding)
    """

    total_loss = 0

    for key in batch:
        # Get images and embeddings
        curr_img_embedding = batch[key]
        curr_images = [pair[0] for pair in curr_img_embedding]
        curr_embeddings = [pair[1] for pair in curr_img_embedding]

        # We iterate over anchors for this class
        for idx, (anchor_img, anchor_embedding) in enumerate(zip(curr_images, curr_embeddings)):

            # First, select anchor
            hardest_anchor = anchor_img
            raise NotImplementedError


    return total_loss
