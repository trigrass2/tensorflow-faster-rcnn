import matplotlib.pyplot as plt


def plot_image_with_bbox(img, bbox, scores, labels, cls_name):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, aspect='equal')
    for box, score, label in zip(bbox, scores, labels):
        ax.add_patch(
            plt.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                fill=False, edgecolor='red', linewidth=3.5)
        )
        ax.text(box[0], box[1] - 2,
                '{:s} {:.3f}'.format(cls_name[label], score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
