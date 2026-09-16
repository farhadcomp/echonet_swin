"""EchoNet-Dynamic Dataset.

Changes vs original EchoNet echo.py
-------------------------------------
FIX-1  ED-frame temporal jitter during training (±6 frames around the
        annotated End-Diastolic frame) for temporal diversity.
FIX-2  Safety clamp guards both ends (negative AND overflow) after jitter.
FIX-3  All-clips inference uses ED-frame anchoring + cycle-stride windows
        instead of noisy pixel-sum peak detection.
FIX-4  Fixed silent key-construction bug: self.fnames already contains
        '.avi', so appending it again caused every ED-frame lookup to miss
        and fall back silently to random starts.
FIX-5  Removed the now-unused scipy.signal import from the clips path.

AREA-TARGET  (new, opt-in via add_area_target=True)
        When enabled, __getitem__ returns (video, (ef, area_target)) where
        area_target = [area_ed, area_es, bin_ed, bin_es, valid] for the
        per-bin area-consistency auxiliary task. When disabled (default),
        behaviour is identical to before: returns (video, target).
"""

import collections
import os

import numpy as np
import pandas
import pandas as pd
import skimage.draw
import torch
import torchvision

import echonet


class Echo(torchvision.datasets.VisionDataset):
    """EchoNet-Dynamic Dataset. See module docstring for arg details."""

    def __init__(
        self,
        root=None,
        split="train",
        target_type="EF",
        mean=0.0,
        std=1.0,
        length=16,
        period=2,
        max_length=250,
        clips=1,
        pad=None,
        noise=None,
        augment=False,
        add_mask=True,
        mask_source="gt",
        mask_dir="MaskedVideos",
        add_area_target=False,            # AREA-TARGET: opt-in flag
        dense_clips=False,                # DENSE-EVAL: use every-start all-clips (original EchoNet protocol)
        n_bins=18,                        # AREA-TARGET: temporal bins (T') from backbone
        target_transform=None,
        external_test_location=None,
    ):
        if root is None:
            root = echonet.config.DATA_DIR

        super().__init__(root, target_transform=target_transform)

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.augment = augment
        self.add_mask = add_mask
        self.mask_source = mask_source
        self.mask_dir = mask_dir
        self.add_area_target = add_area_target    # AREA-TARGET
        self.dense_clips = dense_clips            # DENSE-EVAL
        self.n_bins = n_bins                      # AREA-TARGET
        self._aug_print_count = 0
        self.target_transform = target_transform
        self.external_test_location = external_test_location

        self.fnames, self.outcome = [], []

        tracing_csv = os.path.join(self.root, "VolumeTracings.csv")
        if os.path.exists(tracing_csv):
            df_traces = pd.read_csv(tracing_csv)
            df_traces["FileName"] = df_traces["FileName"].apply(
                lambda x: x if x.endswith(".avi") else x + ".avi"
            )
            self.ed_frames = (
                df_traces.groupby("FileName")["Frame"].min().to_dict()
            )
        else:
            print(
                "WARNING: VolumeTracings.csv not found. "
                "Falling back to random clip starts."
            )
            self.ed_frames = {}

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            with open(os.path.join(self.root, "FileList.csv")) as f:
                data = pandas.read_csv(f)
            data["Split"] = data["Split"].map(lambda x: x.upper())

            if self.split != "ALL":
                data = data[data["Split"] == self.split]

            self.header = data.columns.tolist()
            self.fnames = data["FileName"].tolist()
            self.fnames = [
                fn + ".avi" if os.path.splitext(fn)[1] == "" else fn
                for fn in self.fnames
            ]
            self.outcome = data.values.tolist()

            missing = set(self.fnames) - set(
                os.listdir(os.path.join(self.root, "Videos"))
            )
            if len(missing) != 0:
                print(
                    "{} videos could not be found in {}:".format(
                        len(missing), os.path.join(self.root, "Videos")
                    )
                )
                for fn in sorted(missing):
                    print("\t", fn)
                raise FileNotFoundError(
                    os.path.join(self.root, "Videos", sorted(missing)[0])
                )

            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")
                assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

                for line in f:
                    filename, x1, y1, x2, y2, frame = line.strip().split(",")
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    frame = int(frame)
                    key = filename if filename.endswith(".avi") else filename + ".avi"
                    if frame not in self.trace[key]:
                        self.frames[key].append(frame)
                    self.trace[key][frame].append((x1, y1, x2, y2))

            for key in self.frames:
                for frame in self.frames[key]:
                    self.trace[key][frame] = np.array(self.trace[key][frame])

            keep = [len(self.frames[fn]) >= 2 for fn in self.fnames]
            self.fnames = [fn for fn, k in zip(self.fnames, keep) if k]
            self.outcome = [o for o, k in zip(self.outcome, keep) if k]

    # ---------------------------------------------------------------------- #
    # AREA-TARGET helper: GT polygon area at a given frame
    # ---------------------------------------------------------------------- #
    def _gt_area(self, key, fr, h=112, w=112):
        tr = self.trace[key][fr]
        x1, y1, x2, y2 = tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3]
        x = np.concatenate((x1[1:], np.flip(x2[1:])))
        yc = np.concatenate((y1[1:], np.flip(y2[1:])))
        r, c = skimage.draw.polygon(
            np.rint(yc).astype(int), np.rint(x).astype(int), (h, w)
        )
        return float(len(r))

    # ---------------------------------------------------------------------- #
    # __getitem__
    # ---------------------------------------------------------------------- #
    def __getitem__(self, index):

        if self.split == "EXTERNAL_TEST":
            video_path = os.path.join(
                self.external_test_location, self.fnames[index]
            )
        elif self.split == "CLINICAL_TEST":
            video_path = os.path.join(
                self.root, "ProcessedStrainStudyA4c", self.fnames[index]
            )
        else:
            video_path = os.path.join(self.root, "Videos", self.fnames[index])

        video = echonet.utils.loadvideo(video_path).astype(np.float32)

        USE_AUTOCROP = False
        if USE_AUTOCROP:
            spatial_mask = video.sum(axis=(0, 1))
            rows = np.any(spatial_mask, axis=1)
            cols = np.any(spatial_mask, axis=0)
            if rows.any() and cols.any():
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                crop_pad = 5
                rmin = max(0, rmin - crop_pad)
                rmax = min(video.shape[2], rmax + crop_pad + 1)
                cmin = max(0, cmin - crop_pad)
                cmax = min(video.shape[3], cmax + crop_pad + 1)
                cropped = video[:, :, rmin:rmax, cmin:cmax]
                orig_h, orig_w = video.shape[2], video.shape[3]
                t = torch.from_numpy(cropped).permute(1, 0, 2, 3)
                t = torch.nn.functional.interpolate(
                    t, size=(orig_h, orig_w), mode="bilinear", align_corners=False
                )
                video = t.permute(1, 0, 2, 3).numpy()

        if self.noise is not None:
            n_pixels = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(
                n_pixels, round(self.noise * n_pixels), replace=False
            )
            f_idx = ind % video.shape[1]
            ind //= video.shape[1]
            i_idx = ind % video.shape[2]
            ind //= video.shape[2]
            j_idx = ind
            video[:, f_idx, i_idx, j_idx] = 0

        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        c, f, h, w = video.shape
        video_filename = self.fnames[index]

        # ---- mask channel (unchanged) ----
        if self.mask_source == "zero":
            mask_channel = np.zeros((1, f, h, w), np.float32)
        elif self.mask_source == "predicted":
            masked_path = os.path.join(
                self.root, self.mask_dir, self.fnames[index]
            )
            if os.path.exists(masked_path):
                m_raw = echonet.utils.loadvideo(masked_path).astype(np.float32)
                mask_channel = (m_raw[0:1] > 127).astype(np.float32)
                mask_channel = (mask_channel - 0.07) / 0.26
            else:
                mask_channel = np.zeros((1, f, h, w), np.float32)
        elif self.mask_source == "temporal":
            key = video_filename
            emphasis = np.zeros(f, np.float32)
            if key in self.frames and len(self.frames[key]) >= 2:
                ed_fr = int(min(self.frames[key]))
                es_fr = int(max(self.frames[key]))
                sigma = 3.0
                t = np.arange(f)
                emphasis = (np.exp(-0.5 * ((t - ed_fr) / sigma) ** 2)
                            + np.exp(-0.5 * ((t - es_fr) / sigma) ** 2)).astype(np.float32)
                if emphasis.max() > 0:
                    emphasis = emphasis / emphasis.max()
            mask_channel = np.broadcast_to(
                emphasis[None, :, None, None], (1, f, h, w)
            ).astype(np.float32).copy()
            mask_channel = mask_channel - 0.1
        else:
            key = video_filename
            if key in self.frames and len(self.frames[key]) >= 2:
                def make_mask(trace):
                    x1, y1, x2, y2 = trace[:, 0], trace[:, 1], trace[:, 2], trace[:, 3]
                    x = np.concatenate((x1[1:], np.flip(x2[1:])))
                    y_c = np.concatenate((y1[1:], np.flip(y2[1:])))
                    r, c_idx = skimage.draw.polygon(
                        np.rint(y_c).astype(int),
                        np.rint(x).astype(int),
                        (h, w),
                    )
                    m = np.zeros((h, w), np.float32)
                    m[r, c_idx] = 1.0
                    return m

                frame_a = self.frames[key][0]
                frame_b = self.frames[key][-1]
                mask_a = make_mask(self.trace[key][frame_a])
                mask_b = make_mask(self.trace[key][frame_b])

                if mask_a.sum() >= mask_b.sum():
                    ed_frame, mask_ed = frame_a, mask_a
                    es_frame, mask_es = frame_b, mask_b
                else:
                    ed_frame, mask_ed = frame_b, mask_b
                    es_frame, mask_es = frame_a, mask_a

                mask_volume = np.zeros((f, h, w), np.float32)
                t0 = min(ed_frame, es_frame)
                t1 = max(ed_frame, es_frame)
                m0 = mask_ed if ed_frame < es_frame else mask_es
                m1 = mask_es if ed_frame < es_frame else mask_ed

                for t in range(f):
                    if t <= t0:
                        mask_volume[t] = m0
                    elif t >= t1:
                        mask_volume[t] = m1
                    else:
                        alpha = (t - t0) / (t1 - t0)
                        mask_volume[t] = (1 - alpha) * m0 + alpha * m1

                mask_channel = mask_volume[np.newaxis, :, :, :]
                mask_channel = (mask_channel - 0.07) / 0.26
            else:
                mask_channel = np.zeros((1, f, h, w), np.float32)

        if self.add_mask:
            video = np.concatenate([video, mask_channel], axis=0)

        c, f, h, w = video.shape

        if self.length is None:
            length = f // self.period
        else:
            length = self.length

        if self.max_length is not None:
            length = min(length, self.max_length)

        if f < length * self.period:
            video = np.concatenate(
                (
                    video,
                    np.zeros(
                        (c, length * self.period - f, h, w), video.dtype
                    ),
                ),
                axis=1,
            )
            c, f, h, w = video.shape

        # ---- clip start ----
        if self.clips == "all":
            cycle_len = length * self.period
            if self.dense_clips:
                # DENSE-EVAL: original EchoNet protocol — every possible start.
                # This matches the R(2+1)D baseline and published numbers, which
                # average over ALL overlapping clips (dense test-time augmentation).
                start = list(np.arange(max(1, f - (length - 1) * self.period)))
                if len(start) == 0:
                    start = [max(0, f - cycle_len)]
            elif hasattr(self, "ed_frames") and video_filename in self.ed_frames:
                ed_anchor = int(self.ed_frames[video_filename])
                ed_anchor = max(0, min(ed_anchor, max(0, f - cycle_len)))
                stride = self.period * 4
                start = list(np.arange(ed_anchor, f - (length - 1) * self.period, stride))
                if len(start) == 0:
                    start = [max(0, f - cycle_len)]
            else:
                stride = self.period * 4
                start = list(np.arange(0, f - (length - 1) * self.period, stride))
                if len(start) == 0:
                    start = [max(0, f - cycle_len)]
        else:
            if hasattr(self, "ed_frames") and video_filename in self.ed_frames:
                ed_start = int(self.ed_frames[video_filename])
            else:
                ed_start = np.random.randint(
                    0, max(1, f - length * self.period + 1)
                )

            if self.split == "TRAIN":
                span = self.period * (length - 1)
                key = video_filename
                if key in self.frames and len(self.frames[key]) >= 2:
                    ed_fr = int(min(self.frames[key]))
                    es_fr = int(max(self.frames[key]))
                    lo_needed = max(ed_fr, es_fr)
                    hi_needed = min(ed_fr, es_fr)
                    valid_lo = max(0, lo_needed - span)
                    valid_hi = min(hi_needed, max(0, f - length * self.period))
                    if valid_hi >= valid_lo:
                        start_frame = np.random.randint(valid_lo, valid_hi + 1)
                    else:
                        mid = (ed_fr + es_fr) // 2
                        start_frame = max(0, min(mid - span // 2,
                                                 max(0, f - length * self.period)))
                else:
                    start_frame = np.random.randint(0, max(1, f - length * self.period + 1))
            else:
                start_frame = ed_start

            start_frame = max(0, min(start_frame, max(0, f - length * self.period)))
            start = [start_frame]

        # ------------------------------------------------------------------ #
        # AREA-TARGET: compute ED/ES GT areas and their temporal-bin indices.
        # Only for single-clip mode (training/val), only when enabled.
        # bin = ((gt_frame - start_frame) / period) // (length / n_bins)
        # With length=36, n_bins=18 -> frames_per_bin = 2.
        # ------------------------------------------------------------------ #
        area_ed = area_es = 0.0
        bin_ed = bin_es = -1
        valid_area = 0.0
        if self.add_area_target and self.clips != "all":
            key = video_filename
            if key in self.frames and len(self.frames[key]) >= 2:
                sf = start[0]
                ed_fr = int(min(self.frames[key]))
                es_fr = int(max(self.frames[key]))
                pos_ed = (ed_fr - sf) / self.period      # index in 0..length-1
                pos_es = (es_fr - sf) / self.period
                frames_per_bin = max(1.0, length / float(self.n_bins))
                if 0 <= pos_ed < length and 0 <= pos_es < length:
                    area_ed = self._gt_area(key, ed_fr)
                    area_es = self._gt_area(key, es_fr)
                    bin_ed = min(int(pos_ed // frames_per_bin), self.n_bins - 1)
                    bin_es = min(int(pos_es // frames_per_bin), self.n_bins - 1)
                    valid_area = 1.0

        # ---- debug print (unchanged) ----
        if (not self.clips == "all") and self.split == "TRAIN" \
                and getattr(self, "_aug_print_count", 0) < 8:
            key = video_filename
            if key in self.frames and len(self.frames[key]) >= 2:
                ed_fr = int(min(self.frames[key])); es_fr = int(max(self.frames[key]))
                sf = start[0]
#                 print(f"[clipdbg] {key} ED={ed_fr} ES={es_fr} start={sf} "
#                       f"span=[{sf},{sf+self.period*(length-1)}] "
#                       f"ED_in={sf<=ed_fr<=sf+self.period*(length-1)} "
#                       f"ES_in={sf<=es_fr<=sf+self.period*(length-1)}", flush=True)
#                 self._aug_print_count += 1

        # ---- targets (EF etc., unchanged) ----
        target = []
        for t in self.target_type:
            key = video_filename
            if t == "Filename":
                target.append(video_filename)
            elif t == "LargeIndex":
                target.append(int(self.frames[key][-1]))
            elif t == "SmallIndex":
                target.append(int(self.frames[key][0]))
            elif t == "LargeFrame":
                target.append(video[:, self.frames[key][-1], :, :])
            elif t == "SmallFrame":
                target.append(video[:, self.frames[key][0], :, :])
            elif t in ["LargeTrace", "SmallTrace"]:
                if t == "LargeTrace":
                    tr = self.trace[key][self.frames[key][-1]]
                else:
                    tr = self.trace[key][self.frames[key][0]]
                x1, y1, x2, y2 = tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y_coord = np.concatenate((y1[1:], np.flip(y2[1:])))
                r, c_idx = skimage.draw.polygon(
                    np.rint(y_coord).astype(int),
                    np.rint(x).astype(int),
                    (video.shape[2], video.shape[3]),
                )
                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                mask[r, c_idx] = 1
                target.append(mask)
            else:
                if self.split in ("CLINICAL_TEST", "EXTERNAL_TEST"):
                    target.append(np.float32(0))
                else:
                    target.append(
                        np.float32(self.outcome[index][self.header.index(t)])
                    )

        if target:
            if isinstance(target[0], (np.ndarray, str)):
                target = tuple(target) if len(target) > 1 else target[0]
            else:
                target = (
                    np.array(target, dtype=np.float32)
                    if len(target) > 1
                    else target[0]
                )
            if self.target_transform is not None:
                target = self.target_transform(target)

        # ---- extract clips ----
        video = tuple(
            video[:, s + self.period * np.arange(length), :, :]
            for s in start
        )
        if self.clips == 1:
            video = video[0]
        else:
            video = np.stack(video)

        if self.pad is not None:
            c, l, h, w = video.shape
            temp = np.zeros(
                (c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype
            )
            temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i: i + h, j: j + w]

        if self.split == "TRAIN" and self.augment:
            video = np.ascontiguousarray(video, dtype=np.float32)
            import random
            # video shape here: (C, L, H, W)

            # (a) Random horizontal flip — EF is area-based, flip-invariant; safe.
            if random.random() < 0.5:
                video = video[:, :, :, ::-1]

            # (b) Intensity scale (gain variation across ultrasound machines).
            if random.random() < 0.8:
                scale = random.uniform(0.9, 1.1)
                video = video * scale

            # (c) Additive brightness shift.
            if random.random() < 0.5:
                video = video + random.uniform(-0.1, 0.1)

            # (d) Small random rotation (probe-angle variation), applied per-clip
            #     to all frames identically. Keep small (+-8 deg) to avoid
            #     distorting apparent chamber area too much.
            if random.random() < 0.5:
                import scipy.ndimage
                angle = random.uniform(-8.0, 8.0)
                # rotate spatial dims (H,W)=(2,3), reshape=False keeps size,
                # order=1 bilinear, fill with the normalized-mean (~0).
                video = scipy.ndimage.rotate(
                    video, angle, axes=(2, 3), reshape=False, order=1, mode="constant", cval=0.0
                )

            video = np.ascontiguousarray(video, dtype=np.float32)
            video = np.clip(video, -5.0, 5.0).astype(np.float32)

        # ------------------------------------------------------------------ #
        # AREA-TARGET: package and return.
        # When enabled, return (video, (ef_target, area_target)).
        # When disabled, return (video, target) exactly as before.
        # ------------------------------------------------------------------ #
        if self.add_area_target:
            area_target = np.array(
                [area_ed, area_es, float(bin_ed), float(bin_es), valid_area],
                dtype=np.float32,
            )
            return video, (target, area_target)

        return video, target

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)


def _defaultdict_of_lists():
    return collections.defaultdict(list)