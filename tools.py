import matplotlib.pyplot as plt


def plot_image_with_bbox(img, bbox):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, aspect='equal')
    for box in bbox:
        ax.add_patch(
            plt.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                fill=False, edgecolor='red', linewidth=3.5)
        )

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
