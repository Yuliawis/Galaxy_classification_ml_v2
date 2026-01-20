import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

TRAIN_CSV = r"MyCoords_training.csv"
TRAIN_IMG_ROOT = r"training_images"

ELL_DIR = "ELLIPTICAL"
SPI_DIR = "SPIRAL"

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
FINE_TUNE_EPOCHS = 8

VAL_SPLIT = 0.15
TEST_SPLIT = 0.20

DROP_UNCERTAIN = True

MIN_DIGITS = 8

tf.keras.utils.set_random_seed(SEED)

_digit_re = re.compile(r"\d+")

def objid_to_str(x) -> str:
    if pd.isna(x):
        return ""
    try:
        return str(int(x))
    except Exception:
        pass
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def extract_candidate_ids(filename_stem: str):
    chunks = _digit_re.findall(filename_stem)
    chunks = [c for c in chunks if len(c) >= MIN_DIGITS]
    if not chunks:
        return "", []
    best = max(chunks, key=len)
    return best, chunks

def build_image_index(img_root: str):
    img_root = Path(img_root)
    idx_best = {}
    idx_all = {}

    total_files = 0
    no_digits = 0

    for cls_name in [ELL_DIR, SPI_DIR]:
        cls_path = img_root / cls_name
        if not cls_path.exists():
            raise FileNotFoundError(f"Missing folder: {cls_path}")

        for f in cls_path.rglob("*"):
            if not f.is_file():
                continue
            total_files += 1

            stem = f.stem
            best, chunks = extract_candidate_ids(stem)
            if not best:
                no_digits += 1
                continue

            idx_best[best] = str(f)
            for c in chunks:
                idx_all[c] = str(f)

    print(f"\n[INDEX] {img_root}")
    print(f"  total image files scanned: {total_files}")
    print(f"  files with no digit chunk >= {MIN_DIGITS}: {no_digits}")
    print(f"  unique best-IDs indexed: {len(idx_best)}")
    print(f"  unique all-candidates indexed: {len(idx_all)}")

    if len(idx_best) > 0:
        sample_keys = list(idx_best.keys())[:5]
        print("  sample extracted IDs:", sample_keys)

    return idx_best, idx_all

def csv_label_from_votes(row) -> int:
    sp = float(row["SPIRAL"])
    el = float(row["ELLIPTICAL"])
    un = float(row["UNCERTAIN"])

    m = max(sp, el, un)

    if DROP_UNCERTAIN and abs(un - m) < 1e-12:
        return -1

    ties = sum([abs(sp - m) < 1e-12, abs(el - m) < 1e-12, abs(un - m) < 1e-12])
    if ties > 1:
        return -1

    return 1 if abs(sp - m) < 1e-12 else 0

def build_paths_labels_from_csv(csv_path: str, img_root: str):
    df = pd.read_csv(csv_path)

    required = ["OBJID", "SPIRAL", "ELLIPTICAL", "UNCERTAIN"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")

    idx_best, idx_all = build_image_index(img_root)

    paths, labels = [], []
    not_found = 0
    skipped = 0
    found_by_best = 0
    found_by_any = 0

    print(f"\n[CSV] {Path(csv_path).name} rows: {len(df)}")
    print("  sample OBJIDs:", [objid_to_str(x) for x in df["OBJID"].head(5).tolist()])

    for _, row in df.iterrows():
        objid = objid_to_str(row["OBJID"])
        if not objid:
            skipped += 1
            continue

        path = idx_best.get(objid)
        if path is not None:
            found_by_best += 1
        else:
            path = idx_all.get(objid)
            if path is not None:
                found_by_any += 1

        if path is None:
            not_found += 1
            continue

        y = csv_label_from_votes(row)
        if y == -1:
            skipped += 1
            continue

        paths.append(path)
        labels.append(y)

    paths = np.array(paths, dtype=object)
    labels = np.array(labels, dtype=np.int32)

    print(f"\n[MATCH RESULT] {Path(csv_path).name} vs {img_root}")
    print(f"  matched by best-ID: {found_by_best}")
    print(f"  matched by any-chunk: {found_by_any}")
    print(f"  not found: {not_found}")
    print(f"  skipped (uncertain/ties/blank): {skipped}")
    print(f"  usable samples: {len(paths)}")
    if len(paths) > 0:
        print(f"  class counts: {ELL_DIR}={(labels==0).sum()} | {SPI_DIR}={(labels==1).sum()}")

    if len(paths) == 0:
        root = Path(img_root)
        some = []
        for cls_name in [ELL_DIR, SPI_DIR]:
            cls_path = root / cls_name
            for f in cls_path.glob("*"):
                if f.is_file():
                    some.append(f.name)
                if len(some) >= 10:
                    break
            if len(some) >= 10:
                break

        raise ValueError(
            "No usable samples after matching + filtering.\n"
            "Most likely your filenames do not contain OBJID as a digit chunk.\n"
            f"Examples of image filenames found: {some}\n"
            "Fix: rename images to include OBJID, or adjust extract_candidate_ids() for your filename pattern."
        )

    return paths, labels

def make_tf_dataset(paths, labels, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_img(path, y):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)
        return img, tf.cast(y, tf.float32)

    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ])

    if training:
        ds = ds.shuffle(min(len(paths), 5000), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda x, y: (aug(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# loadind data + splitting (20% test)

all_paths, all_labels = build_paths_labels_from_csv(TRAIN_CSV, TRAIN_IMG_ROOT)

# 20% test from full dataset
train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    all_paths, all_labels,
    test_size=TEST_SPLIT,
    random_state=SEED,
    stratify=all_labels
)

# validation from remaining 80%
X_train, X_val, y_train, y_val = train_test_split(
    train_val_paths, train_val_labels,
    test_size=VAL_SPLIT,
    random_state=SEED,
    stratify=train_val_labels
)

print("\n[SPLIT]")
print(f"  total: {len(all_paths)}")
print(f"  train: {len(X_train)}")
print(f"  val:   {len(X_val)}")
print(f"  test:  {len(test_paths)}")

train_ds = make_tf_dataset(X_train, y_train, training=True)
val_ds   = make_tf_dataset(X_val, y_val, training=False)
test_ds  = make_tf_dataset(test_paths, test_labels, training=False)

# model

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = tf.keras.layers.Rescaling(1./255)(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)

base = tf.keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False

x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs, name="galaxy_binary_cnn")
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"), tf.keras.metrics.AUC(name="auc")]
)
model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_galaxy_cnn.keras",
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
]

# training
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# fine tuning
print("\n--- Fine-tuning ---")
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"), tf.keras.metrics.AUC(name="auc")]
)
model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS, callbacks=callbacks)

# testing
print("\nFinal evaluation on testing (20% holdout)")
metrics = model.evaluate(test_ds, verbose=1)
print(dict(zip(model.metrics_names, metrics)))

y_prob = model.predict(test_ds, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

cm = confusion_matrix(test_labels, y_pred)
print("\nConfusion Matrix (rows=true, cols=pred):")
print(cm)

print("\nClassification Report:")
print(classification_report(test_labels, y_pred, target_names=[ELL_DIR, SPI_DIR], digits=4))

model.save("final_galaxy_cnn_fr.keras")
print("\nSaved")
