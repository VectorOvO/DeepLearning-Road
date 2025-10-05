def test_dl(dataloader):
    # 显示几张数据集图片
    import matplotlib.pyplot as plt
    images, labels = next(iter(dataloader))
    plt.figure(figsize=(10, 4))
    for i in range(8):
        plt.subplot(2, 4, i+1)
        plt.imshow(images[i][0], cmap="gray")
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")
    plt.show()