import os
import json
from glob import glob


def with_id(basename: str, shard_id: int) -> str:
    """Get a new basename with the given shard_id.

    Args:
        basename (str): Old basename of file.
        shard_id (int): New shard ID.

    Returns:
        str: New basename of file.
    """
    parts = basename.split(".")
    parts[1] = f"{shard_id:04}"
    return ".".join(parts)


def merge_shard_groups(
    root: str = "./vae_mds",
) -> None:
    """Merge ephemeral sub-datasets created in parallel into one dataset.

    Args:
        root (str): Root directory.
    """
    pattern = os.path.join(root, "*")
    subdirs = sorted(glob(pattern))
    shard_id = 0
    infos = []
    for subdir in subdirs:
        index_filename = os.path.join(subdir, "data/index.json")
        if not os.path.isfile(index_filename):
            continue
        obj = json.load(open(index_filename))
        for info in obj["shards"]:
            old_basename = info["raw_data"]["basename"]
            new_basename = with_id(old_basename, shard_id)
            info["raw_data"]["basename"] = new_basename

            if info["zip_data"] is not None:
                old_basename = info["zip_data"]["basename"]
                new_basename = with_id(old_basename, shard_id)
                info["zip_data"]["basename"] = new_basename

            old_filename = os.path.join(subdir, "data/" + old_basename)
            new_filename = os.path.join(root, new_basename)
            assert not os.rename(old_filename, new_filename)

            shard_id += 1
            infos.append(info)

        assert not os.remove(index_filename)
        # assert not os.rmdir(subdir)

    index_filename = os.path.join(root, "index.json")
    obj = {
        "version": 2,
        "shards": infos,
    }
    text = json.dumps(obj, sort_keys=True)
    with open(index_filename, "w") as out:
        out.write(text)


merge_shard_groups()