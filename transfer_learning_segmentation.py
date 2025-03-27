

import json
import torch
import datasets
import requests
import evaluate
import numpy as np
import huggingface_hub
from PIL import Image
import albumentations as A
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Any
from dataclasses import dataclass
from datasets import load_dataset
import matplotlib.patches as mpatches
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset, DataLoader



from transformers import (
    MaskFormerImageProcessor,
    AutoImageProcessor,
    MaskFormerForInstanceSegmentation,
)

torch.manual_seed(42)

# it seems that we need to login to huggingface to have access to the dataset segments/sidewalk-semantic
huggingface_hub.notebook_login()


hf_dataset_id = "segments/sidewalk-semantic"
dataset = load_dataset(hf_dataset_id)

dataset = dataset.shuffle(seed=1)
dataset = dataset["train"].train_test_split(test_size=0.2)
train_ds, test_ds = dataset["train"], dataset["test"]

dataset = train_ds.train_test_split(test_size=0.05)
train_ds, val_ds = dataset["train"], dataset["test"]

filename = "id2label.json"
id2label = json.load(
    open(hf_hub_download(hf_dataset_id, filename, repo_type="dataset"), "r")
)
id2label = {int(k): v for k, v in id2label.items()}
print(id2label)


example = train_ds[0]
print(example)
segmentation_map = np.array(example["label"])
image_array = np.array(example["pixel_values"])
print(
    f"Shape : Image: {image_array.shape} - Segmentation map: {segmentation_map.shape}"
)

print(segmentation_map)


def show_samples(dataset: datasets.Dataset, n: int = 5):
    """
    Displays 'n' samples from the dataset.
    ----
    Args:
      - dataset: The dataset which should contain 'pixel_values' and 'label' in its items.
      - n (int): Number of samples to display.

    """
    if n > len(dataset):
        raise ValueError("n is larger than the dataset size")

    fig, axs = plt.subplots(n, 2, figsize=(10, 5 * n))

    for i in range(n):
        sample = dataset[i]
        image, label = np.array(sample["pixel_values"]), sample["label"]

        axs[i, 0].imshow(image)
        axs[i, 0].set_title("Image")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(image)
        axs[i, 1].imshow(label, cmap="nipy_spectral", alpha=0.5)
        axs[i, 1].set_title("Segmentation Map")
        axs[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

show_samples(train_ds, n=4)


preprocessor = MaskFormerImageProcessor(
    ignore_index=0,
    do_reduce_labels=False,
    do_resize=False,
    do_rescale=False,
    do_normalize=False,
)
ade_mean = np.array([123.675, 116.280, 103.530]) / 255
ade_std = np.array([58.395, 57.120, 57.375]) / 255

train_transform = A.Compose(
    [
        A.RandomCrop(width=512, height=512),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=ade_mean, std=ade_std),
    ]
)

test_transform = A.Compose(
    [
        A.Resize(width=512, height=512),
        A.Normalize(mean=ade_mean, std=ade_std),
    ]
)


@dataclass
class SegmentationDataInput:
    original_image: np.ndarray
    transformed_image: np.ndarray
    original_segmentation_map: np.ndarray
    transformed_segmentation_map: np.ndarray


class SemanticSegmentationDataset(Dataset):
    def __init__(self, dataset: datasets.Dataset, transform: Any) -> None:
        """
        Dataset for Semantic Segmentation.
        ----
        Args:
          - dataset: A dataset containing images and segmentation maps.
          - transform: A transformation function to apply to the images and segmentation maps.
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        sample = self.dataset[idx]
        original_image = np.array(sample["pixel_values"])
        original_segmentation_map = np.array(sample["label"])

        transformed = self.transform(
            image=original_image, mask=original_segmentation_map
        )
        transformed_image = transformed["image"].transpose(
            2, 0, 1
        )  # Transpose to channel-first format
        transformed_segmentation_map = transformed["mask"]

        return SegmentationDataInput(
            original_image=original_image,
            transformed_image=transformed_image,
            original_segmentation_map=original_segmentation_map,
            transformed_segmentation_map=transformed_segmentation_map,
        )


def collate_fn(batch: SegmentationDataInput) -> dict:
    original_images = [sample.original_image for sample in batch]
    transformed_images = [sample.transformed_image for sample in batch]
    original_segmentation_maps = [sample.original_segmentation_map for sample in batch]
    transformed_segmentation_maps = [
        sample.transformed_segmentation_map for sample in batch
    ]

    preprocessed_batch = preprocessor(
        transformed_images,
        segmentation_maps=transformed_segmentation_maps,
        return_tensors="pt",
    )

    preprocessed_batch["original_images"] = original_images
    preprocessed_batch["original_segmentation_maps"] = original_segmentation_maps

    return preprocessed_batch

train_dataset = SemanticSegmentationDataset(train_ds, transform=train_transform)
val_dataset = SemanticSegmentationDataset(val_ds, transform=train_transform)
test_dataset = SemanticSegmentationDataset(test_ds, transform=test_transform)

# Prepare Dataloaders
train_dataloader = DataLoader(
    train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
)
test_dataloader = DataLoader(
    test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn
)




sample = next(iter(train_dataloader))
print(
    {
        key: value[0].shape if isinstance(value, list) else value.shape
        for key, value in sample.items()
    }
)


def denormalize_image(image, mean, std):
    """
    Denormalizes a normalized image.
    ----
    Args:
     - image (numpy.ndarray): The normalized image.
     - mean (list or numpy.ndarray): The mean used for normalization.
     - std (list or numpy.ndarray): The standard deviation used for normalization.

    """
    unnormalized_image = (image * std[:, None, None]) + mean[:, None, None]
    unnormalized_image = (unnormalized_image * 255).numpy().astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    return unnormalized_image


denormalized_image = denormalize_image(sample["pixel_values"][0], ade_mean, ade_std)
pil_image = Image.fromarray(denormalized_image)
pil_image



labels = [id2label[label] for label in sample["class_labels"][0].tolist()]
print(labels)

def visualize_mask(sample, labels, label_name):
    print(f"Category: {label_name}")
    idx = labels.index(label_name)

    visual_mask = (sample["mask_labels"][0][idx].bool().numpy() * 255).astype(np.uint8)
    return Image.fromarray(visual_mask)

visualize_mask(sample, labels, labels[0])


processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
model = MaskFormerForInstanceSegmentation.from_pretrained(
    "facebook/maskformer-swin-base-coco"
)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image

inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

predicted_semantic_map = processor.post_process_semantic_segmentation(
    outputs, target_sizes=[image.size[::-1]]
)[0]
predicted_semantic_map

num_classes = len(np.unique(predicted_semantic_map))
cmap = plt.cm.get_cmap("hsv", num_classes)

overlay = np.zeros(
    (predicted_semantic_map.shape[0], predicted_semantic_map.shape[1], 4)
)

for i, unique_value in enumerate(np.unique(predicted_semantic_map)):
    overlay[predicted_semantic_map == unique_value, :3] = cmap(i)[:3]
    overlay[predicted_semantic_map == unique_value, 3] = 0.5  # 50% transparency

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(image)
ax.imshow(overlay, interpolation="nearest", alpha=0.9)
plt.axis("off")

plt.show()


predicted_semantic_map = processor.post_process_panoptic_segmentation(
    outputs, target_sizes=[image.size[::-1]]
)[0]
predicted_semantic_map

num_instances = len(predicted_semantic_map["segments_info"])
colors = plt.cm.get_cmap("viridis", num_instances)
overlay = np.zeros((np.array(image).shape[0], np.array(image).shape[1], 4))

for i, info in enumerate(predicted_semantic_map["segments_info"]):
    mask = predicted_semantic_map["segmentation"] == info["id"]
    color = colors(i)
    overlay[mask, :3] = color[:3]  # RGB channels
    overlay[mask, 3] = 0.5  # Alpha channel for transparency

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(np.array(image))
ax.imshow(overlay, interpolation="nearest", alpha=0.9)
plt.axis("off")

handles = [
    mpatches.Patch(
        color=colors(i)[:3], label=f"{model.config.id2label[info['label_id']]}-{i}"
    )
    for i, info in enumerate(predicted_semantic_map["segments_info"])
]
plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")

plt.show()


model = MaskFormerForInstanceSegmentation.from_pretrained(
    "facebook/maskformer-swin-base-ade", id2label=id2label, ignore_mismatched_sizes=True
)

sample = next(iter(train_dataloader))
outputs = model(
    pixel_values=sample["pixel_values"],
    pixel_mask=sample["pixel_mask"],
    class_labels=sample["class_labels"],
    mask_labels=sample["mask_labels"],
)
print(outputs.loss)

# pixel level module contains both the backbone and the pixel decoder
for param in model.model.pixel_level_module.parameters():
    param.requires_grad = False

# Confirm that the parameters are correctly frozen
for name, param in model.model.pixel_level_module.named_parameters():
    assert not param.requires_grad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metric = evaluate.load("mean_iou")


def evaluate_model(
    model: MaskFormerForInstanceSegmentation,
    dataloader: DataLoader,
    preprocessor: AutoImageProcessor,
    metric: Any,
    id2label: dict,
    max_batches=None,
):
    """
    Evaluates the given model using the specified dataloader and computes the mean Intersection over Union (IoU).
    ----
    Args:
      - model (MaskFormerForInstanceSegmentation): The trained model to be evaluated.
      - dataloader (DataLoader): DataLoader containing the dataset for evaluation.
      - preprocessor (AutoImageProcessor): The preprocessor used for post-processing the model outputs.
      - metric (Any): Metric instance used for calculating IoU.
      - id2label (dict): Dictionary mapping class ids to their corresponding labels.
      - max_batches (int, optional): Maximum number of batches to evaluate. If None, evaluates on the entire validation dataset.

    Returns:
    float: The mean IoU calculated over the specified number of batches.
    """
    model.eval()
    running_iou = 0
    num_batches = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            if max_batches and idx >= max_batches:
                break

            pixel_values = batch["pixel_values"].to(device)
            outputs = model(pixel_values=pixel_values)

            original_images = batch["original_images"]
            target_sizes = [
                (image.shape[0], image.shape[1]) for image in original_images
            ]

            predicted_segmentation_maps = (
                preprocessor.post_process_semantic_segmentation(
                    outputs, target_sizes=target_sizes
                )
            )

            ground_truth_segmentation_maps = batch["original_segmentation_maps"]
            metric.add_batch(
                references=ground_truth_segmentation_maps,
                predictions=predicted_segmentation_maps,
            )

            running_iou += metric.compute(num_labels=len(id2label), ignore_index=0)[
                "mean_iou"
            ]
            num_batches += 1

    mean_iou = running_iou / num_batches
    return mean_iou


def train_model(
    model: MaskFormerForInstanceSegmentation,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    preprocessor: AutoImageProcessor,
    metric: AutoImageProcessor,
    id2label: dict,
    num_epochs=100,
    learning_rate=5e-5,
    log_interval=100,
):
    """
    Trains the MaskFormer model for semantic segmentation over a specified number of epochs and evaluates it on a validation set.
    ----
    Args:
      - model (MaskFormerForInstanceSegmentation): The model to be trained.
      - train_dataloader (DataLoader): DataLoader for the training data.
      - val_dataloader (DataLoader): DataLoader for the validation data.
      - preprocessor (AutoImageProcessor): The preprocessor used for preparing the data.
      - metric (Any): Metric instance used for calculating performance metrics.
      - id2label (dict): Dictionary mapping class IDs to their corresponding labels.
      - num_epochs (int): Number of epochs to train the model.
      - learning_rate (float): Learning rate for the optimizer.
      - log_interval (int): Interval (in number of batches) at which to log training progress.

    """
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"Current epoch: {epoch+1}/{num_epochs}")
        model.train()

        running_loss = 0.0
        num_samples = 0

        for idx, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()

            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )

            loss = outputs.loss
            loss.backward()

            batch_size = batch["pixel_values"].size(0)
            running_loss += loss.item()
            num_samples += batch_size

            if idx % log_interval == 0 and idx > 0:
                print(f"Iteration {idx} - loss: {running_loss/num_samples}")

            optimizer.step()
        val_mean_iou = evaluate_model(
            model, val_dataloader, preprocessor, metric, id2label, max_batches=6
        )
        print(f"Validation Mean IoU: {val_mean_iou}")

train_model(
    model,
    train_dataloader,
    val_dataloader,
    preprocessor,
    metric,
    id2label,
    num_epochs=2,
    log_interval=100,
)


test_mean_iou = evaluate_model(model, test_dataloader, preprocessor, metric, id2label)
print(f"Test Mean IoU: {test_mean_iou}")

def show_inference_samples(
    model: MaskFormerForInstanceSegmentation,
    dataloader: DataLoader,
    preprocessor: AutoImageProcessor,
    n: int = 5,
):
    """
    Displays 'n' samples from the dataloader with model inference results.
    ----
    Args:
     - model: The trained model.
     - dataloader: DataLoader containing the dataset for inference.
     - preprocessor: The preprocessor used for post-processing the outputs.
     - n (int): Number of samples to display.

    """
    model.to(device)
    model.eval()

    if n > len(dataloader.dataset):
        raise ValueError("n is larger than the dataset size")

    fig, axs = plt.subplots(n, 2, figsize=(10, 5 * n))

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n:
                break

            pixel_values = batch["pixel_values"].to(device)
            outputs = model(pixel_values=pixel_values)

            original_images = batch["original_images"]
            target_sizes = [
                (image.shape[0], image.shape[1]) for image in original_images
            ]
            predicted_segmentation_maps = (
                preprocessor.post_process_semantic_segmentation(
                    outputs, target_sizes=target_sizes
                )
            )

            ground_truth_segmentation_maps = batch["original_segmentation_maps"]

            # Assuming original_images are numpy arrays and already in the right format
            image = original_images[i]
            ground_truth_map = ground_truth_segmentation_maps[i]
            predicted_map = predicted_segmentation_maps[i]

            axs[i, 0].imshow(image)
            axs[i, 0].imshow(ground_truth_map, cmap="nipy_spectral", alpha=0.5)
            axs[i, 0].set_title("Ground Truth")
            axs[i, 0].axis("off")

            axs[i, 1].imshow(image)
            axs[i, 1].imshow(
                predicted_map.cpu().numpy(), cmap="nipy_spectral", alpha=0.5
            )
            axs[i, 1].set_title("Prediction")
            axs[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

show_inference_samples(model, test_dataloader, preprocessor, n=2)

