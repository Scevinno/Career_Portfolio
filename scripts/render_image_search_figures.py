#########################################################################
# Site-styled figure exports for the image search engine write-up
#########################################################################

# Re-runs the search step of the project's own script against the saved
# index (models/*.p, built by 101_cnn_image_search_engine.py) and renders
# the three figures used in the post, in the website's palette on a
# transparent background.
#
#   1. image_search_results_01.png  - query 1 (red heel) + its eight matches
#   2. image_search_results_02.png  - query 2 (rain boots) + its eight matches
#   3. image_search_distances.png   - what a cosine distance actually means
#
# Run from the dsi-deep-learning env with its DLL directories on PATH:
#   E="$HOME/anaconda3/envs/dsi-deep-learning"
#   export PATH="$E:$E/Library/bin:$E/Library/usr/bin:$E/Scripts:$PATH"
#   python scripts/render_image_search_figures.py

import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

PROJECT = (r"C:\SOREN\Resources\Documents\Coding\Portfolio\Python"
           r"\DSI Deep Learning\Image Search Engine")
OUT_DIR = r"C:\Career_Portfolio\img\posts"

TEXT = "#9fb0c0"        # site --text-dim
BRIGHT = "#eef2f7"      # site --text
TEAL = "#17a589"        # site accent
VIOLET = "#a78bfa"      # secondary accent, as used on project 3
GREY = "#5c6b7f"
LINE = "#2b3646"        # site --line-strong

plt.rcParams["font.family"] = "sans-serif"

img_width = 224
img_height = 224

#########################################################################
# the project's own search, re-run against its saved index
#########################################################################


def preprocess_image(filepath):

    image = load_img(filepath, target_size=(img_width, img_height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    return image


def featurise_image(image):

    return model.predict(image)


model = load_model(PROJECT + r"\models\vgg16_search_engine.h5", compile=False)

filename_store = pickle.load(open(PROJECT + r"\models\filename_store.p", "rb"))
feature_vector_store = pickle.load(open(PROJECT + r"\models\feature_vector_store.p", "rb"))

search_results_n = 8

image_neighbors = NearestNeighbors(n_neighbors=search_results_n, metric="cosine")
image_neighbors.fit(feature_vector_store)


def square(filepath):

    # pad to a square canvas so every result tile is the same shape and the
    # rank/distance captions sit on one line - the source photos vary in
    # aspect ratio. Padding takes the colour of the image's own corners, so
    # a white studio background stays white and a black one stays black.
    image = img_to_array(load_img(filepath)).astype("uint8")
    height, width = image.shape[:2]
    side = max(height, width)

    corners = np.array([image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]])
    fill = np.median(corners, axis=0).astype("uint8")

    canvas = np.tile(fill, (side, side, 1))
    top = (side - height) // 2
    left = (side - width) // 2
    canvas[top:top + height, left:left + width] = image

    return canvas


def search(search_image):

    preprocessed_image = preprocess_image(PROJECT + "\\" + search_image)
    search_feature_vector = featurise_image(preprocessed_image)

    image_distances, image_indices = image_neighbors.kneighbors(search_feature_vector)

    indices = list(image_indices[0])
    distances = list(image_distances[0])
    files = [filename_store[i] for i in indices]

    return files, distances


#########################################################################
# 1 & 2. Query and its eight matches
#########################################################################

def render_results(search_image, out_name, title):

    files, distances = search(search_image)

    fig = plt.figure(figsize=(12, 5.4), dpi=150)
    gs = fig.add_gridspec(2, 5, width_ratios=[1.5, 1, 1, 1, 1],
                          wspace=0.14, hspace=0.34)

    # the search image, held apart from the results
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(load_img(PROJECT + "\\" + search_image))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(TEAL)
        spine.set_linewidth(2)
    ax.set_title("SEARCH IMAGE", color=TEAL, fontsize=11, pad=12, fontweight="bold")

    # the eight nearest catalogue images, in order
    for counter, result_file in enumerate(files):

        ax = fig.add_subplot(gs[counter // 4, counter % 4 + 1])
        ax.imshow(square(PROJECT + "\\" + result_file))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)
            spine.set_linewidth(1)
        ax.set_xlabel("%d   ·   %.3f" % (counter + 1, distances[counter]),
                      color=TEXT, fontsize=10, labelpad=7)

    fig.suptitle(title, color=TEXT, fontsize=14, y=1.02)
    plt.savefig(OUT_DIR + "\\" + out_name, transparent=True, bbox_inches="tight")
    plt.close()

    print(out_name, ["%.3f" % d for d in distances])


render_results("search_image_01.jpg", "image_search_results_01.png",
               "Eight closest images to the red heel, with their cosine distances")

render_results("search_image_02.jpg", "image_search_results_02.png",
               "Eight closest images to the patterned rain boots, with their cosine distances")

#########################################################################
# 3. What a distance actually means
#########################################################################

# for every catalogue image, the distance to its own closest neighbour -
# the reference for judging whether a search result is genuinely close
store_distances, _ = image_neighbors.kneighbors(feature_vector_store, n_neighbors=2)
nearest = store_distances[:, 1]

fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150)

counts, _, _ = ax.hist(nearest, bins=30, color=TEAL, alpha=0.55, edgecolor="none")

# headroom, so the query markers are labelled clear of the bars
ax.set_ylim(0, counts.max() * 1.26)

for value, colour, label in [
        (0.113, BRIGHT, "red heel — best match 0.113"),
        (0.232, VIOLET, "rain boots — best match 0.232")]:
    ax.axvline(value, color=colour, lw=1.6, ls="--")
    ax.text(value + 0.006, ax.get_ylim()[1] * 0.97, label,
            color=colour, fontsize=10.5, va="top")

ax.set_xlabel("cosine distance to nearest neighbour", color=TEXT, fontsize=11, labelpad=9)
ax.set_ylabel("catalogue images", color=TEXT, fontsize=11, labelpad=9)
ax.set_title("Every catalogue image has a nearest neighbour — this is how far away it sits",
             color=TEXT, fontsize=14, pad=16)

ax.tick_params(colors=TEXT, labelsize=10)
ax.grid(axis="y", color=TEXT, alpha=0.12, lw=1)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GREY)

plt.tight_layout()
plt.savefig(OUT_DIR + r"\image_search_distances.png", transparent=True, bbox_inches="tight")
plt.close()

print("distance spread: min %.3f  median %.3f  max %.3f"
      % (nearest.min(), np.median(nearest), nearest.max()))
