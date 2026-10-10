"""All-window RT data using verified frozen parent VAE latents and native transforms.

The parent clip cache's first latent frame is bit-identical to encoding the
observation alone. No random student/teacher state or trainable feature is cached.
"""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from .loopwam_long import LoopWAMLongDataset

class TransportCachedDataset(torch.utils.data.Dataset):
    def __init__(self,manifest_path,latent_dir):
        self.manifest=json.loads(Path(manifest_path).read_text())
        if self.manifest["dataset_scope"]!="full_libero":
            raise ValueError("RT requires full LIBERO")
        stats=json.loads(Path(self.manifest["normalization_path"]).read_text())
        self.transform=LoopWAMLongDataset([],self.manifest["text_cache_dir"],stats)
        self.stats=stats
        self.size=self.manifest["train_windows"]
        directory=Path(latent_dir)
        metadata=json.loads((directory/"metadata.json").read_text())
        if metadata["size"]!=self.size or metadata["latent_shape"]!=[16,3,28,56]:
            raise ValueError("Wrong parent latent cache")
        source=metadata["provenance"]["data_manifest"]
        for key in ("normalization_sha256","content_files","train_episodes","camera_order","action_offsets"):
            if source[key]!=self.manifest[key]:
                raise ValueError(f"Cache/data provenance mismatch: {key}")
        valid=np.memmap(directory/"valid.uint8",mode="r",dtype=np.uint8,shape=(self.size,))
        if not np.all(valid==1):
            raise ValueError("Parent cache must be complete; populate missing windows before training")
        self.latents=np.memmap(directory/"latents.uint16",mode="r",dtype=np.uint16,
                               shape=(self.size,16,3,28,56))
        self.episodes=[]
        self.index=np.empty((self.size,2),dtype=np.int32)
        cursor=0
        for suite in self.manifest["suites"]:
            details=self.manifest["coverage"][suite]
            root=Path(details["dataset_dir"])
            table=pq.read_table(sorted((root/"data").glob("*/*.parquet")),
                               columns=["episode_index","frame_index","action","observation.state"])
            eps=np.asarray(table["episode_index"])
            frames=np.asarray(table["frame_index"])
            acts=np.array(table["action"].to_pylist(),dtype=np.float32)
            states=np.array(table["observation.state"].to_pylist(),dtype=np.float32)
            rows=pq.read_table(sorted((root/"meta/episodes").glob("*/*.parquet")),
                               columns=["episode_index","tasks"]).to_pylist()
            tasks={r["episode_index"]:r["tasks"][0] for r in rows}
            for ep in details["train_episodes"]:
                positions=np.flatnonzero(eps==ep)
                positions=positions[np.argsort(frames[positions],kind="stable")]
                if not np.array_equal(frames[positions],np.arange(len(positions))):
                    raise ValueError("Episode has noncontiguous frames")
                a=acts[positions]; state=states[positions]
                number=len(self.episodes); n=len(a)
                self.index[cursor:cursor+n,0]=number
                self.index[cursor:cursor+n,1]=np.arange(n)
                self.episodes.append((a,state,tasks[ep]))
                cursor+=n
            if cursor!=details["global_index_stop"]:
                raise ValueError("Dataset order differs from parent cache")
        if cursor!=self.size: raise ValueError("Full data coverage mismatch")
        # Keep only ~100MB of tabular data and read first-frame latents on demand.
    def __len__(self): return self.size
    def __getitem__(self,index):
        valid=index>=0; real=max(index,0)
        ep,frame=self.index[real]; a,state,task=self.episodes[ep]
        positions=frame+np.arange(32); pad=positions>=len(a)
        actions=torch.from_numpy(a[np.minimum(positions,len(a)-1)].copy())
        actions[torch.from_numpy(pad),:6]=0
        actions=self.transform.normalizer["action"].forward(actions)
        proprio=self.transform.normalizer["state"].forward(torch.from_numpy(state[frame:frame+1].copy()))
        from .loopwam_long import DEFAULT_PROMPT
        context,mask=self.transform._text(DEFAULT_PROMPT.format(task=task))
        bits=np.array(self.latents[real,:,:1],copy=True).view(np.int16)
        latents=torch.from_numpy(bits).view(torch.bfloat16).float()
        return dict(first_frame_latents=latents,action=actions,proprio=proprio,
            context=context,context_mask=mask,action_is_pad=torch.from_numpy(pad),
            sample_valid=valid,training_index=index)
