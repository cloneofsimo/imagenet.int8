from datasets import load_dataset

dataset = load_dataset("imagenet-1k", split="train")
# check dataset
for i in range(5):
    print(dataset[i]["image"])
