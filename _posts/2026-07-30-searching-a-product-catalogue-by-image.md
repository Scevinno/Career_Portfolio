---
layout: post
title: Searching a Product Catalogue by Image with a Pre-Trained Network
image: "/img/posts/image_search_engine.svg"
tags: [Deep Learning, Computer Vision, Python]
summary: "Three hundred shoe photographs with no tags, no labels and no training run. A network trained on something else turns each image into 512 numbers, and a search becomes a question about distance."
stack: "Python · TensorFlow · Keras · scikit-learn"
metrics:
  - value: "512"
    label: "numbers describe an image"
  - value: "300"
    label: "images in the catalogue"
  - value: "8"
    label: "results per search"
---

Product search usually runs on words, which means somebody has to write them — every item tagged as *red*, *heeled*, *patent*, *platform* before anyone can find it. This project skips the words entirely. Hand it a photograph and it returns the eight images in the catalogue that look most like it. Nothing here is trained and nothing is labelled: a network trained for a different job does the looking, and the search itself is a distance calculation.

---

# Table of Contents

- [00. Project Overview](#00-project-overview)
- [01. Results](#01-results)
- [02. Model Overview](#02-model-overview)
- [03. Data Overview](#03-data-overview)
- [04. Turning Images into Numbers](#04-turning-images-into-numbers)
- [05. Searching the Catalogue](#05-searching-the-catalogue)
- [06. What the Distances Mean](#06-what-the-distances-mean)
- [07. Growth & Next Steps](#07-growth--next-steps)

---

## 00. Project Overview

**Context**

A catalogue of 300 footwear photographs arrives with no tags, no categories and no labels of any kind. Text search has nothing to work with. The question is whether the images alone are enough to find one item from another — whether "show me more like this one" can be answered without anybody describing what *this one* is.

**Actions**

VGG16, a network trained on millions of general photographs, is loaded with its classification layer removed. What remains is the part that learned to look, and it reduces any photograph to 512 numbers. Every image in the catalogue is passed through it once and the results are stored. A search then puts the query photograph through the same process and asks which of the 300 stored vectors sit closest to it by cosine distance.

**Applications**

Anything with a visual catalogue and unreliable tagging. The nearest use is retail — a "more like this" row underneath a product, or a customer photographing something they saw and finding the closest stock match. The same index also answers questions the catalogue owner has, rather than the customer: which listings are near-duplicates of each other, and which product photographs are inconsistent with the rest of the set.

**Growth & Next Steps**

The gap in this project is measurement. Without labels there is no precision score to report, so section 06 does what can be done instead — establishes what a cosine distance means across this catalogue, so a search result can be read against something rather than judged purely by eye.

---

## 01. Results

Two photographs were used as queries. Neither is in the catalogue.

![The eight closest catalogue images to the red heel]({{ "/img/posts/image_search_results_01.png" | relative_url }})

**The red heel returned eight red heels.** All eight results are red and all eight have a heel, with most sharing the platform sole of the query. Nothing in this project was told about colour, heel height or shoe type — the only instruction the model ever received was to describe the image as 512 numbers.

**One result was photographed against a black background.** The eighth match sits on black while the query and every other result sit on white. It still ranked inside the top eight, which suggests the description is carrying the shoe rather than the studio conditions around it. That is the failure mode worth watching in this kind of system, and here it did not dominate.

![The eight closest catalogue images to the patterned rain boots]({{ "/img/posts/image_search_results_02.png" | relative_url }})

**The second query was framed completely differently and still worked.** It is a photograph of someone wearing the boots, legs included, shot from the shins down. The catalogue is entirely product-only shots on white. All eight results are patterned rain boots, and the top three are the same busy multicoloured print as the query.

**The second search sits further away than the first.** Its best match is 0.232 against the red heel's 0.113 — roughly double the distance. The queries differ in how closely they resemble a catalogue photograph, and the distances reflect that: the heel is a studio product shot like the images it is being matched against, and the boots are not.

One thing this section cannot say is how *often* the engine is right. There are no labels, so there is no count of correct and incorrect results — sixteen results judged by eye is what the data supports, and section 06 covers what can be measured instead.

---

## 02. Model Overview

A network trained to classify images has to learn to describe them first. VGG16 was trained on ImageNet to sort photographs into a thousand categories, and the layers that do the sorting sit at the very end. Everything before them is the part that learned what edges, textures, shapes and materials look like — and that part is useful well beyond the thousand categories it was trained for.

```python
# image parameters

img_width = 224
img_height = 224
num_chnnels = 3

# network architecture

vgg = VGG16(input_shape = (img_width, img_height, num_chnnels), include_top = False, pooling = 'avg')
vgg.summary()

model = Model(inputs = vgg.input, outputs = vgg.layers[-1].output)

# save model file

model.save('models/vgg16_search_engine.h5')
```

`include_top = False` is what removes the classification layers. Without it the network returns a thousand category scores; with it, the network returns the feature maps that would have fed those scores.

`pooling = 'avg'` then flattens those maps into a single list. The final convolutional block produces 512 feature maps of 7×7 values each — 25,088 numbers in total — and average pooling replaces each map with its own average, leaving 512. Each of those 512 numbers is roughly "how much of this pattern appears anywhere in this image", which is exactly the right question for a search: it does not matter *where* in the frame the shoe is, only that it is there.

224×224 is not an arbitrary size. It is the input size VGG16 was trained at, and feeding it something else changes what its filters see.

Nothing in this project is trained. No labels are needed, no fitting happens, and the whole thing runs on a laptop CPU.

---

## 03. Data Overview

The catalogue is 300 footwear photographs in a single flat folder, named `footwear_0000.jpg` through to `footwear_0301.jpg`. There are no subfolders, no categories and no labels — which is the point rather than a shortcoming.

| | |
|---|---|
| Catalogue images | 300 |
| Query images | 2, held outside the catalogue |
| Labels | none |
| Image size fed to the network | 224 × 224 |
| Numbers stored per image | 512 |

This is a different starting position from a classification project, where the folder an image sits in *is* its label. Here there is nothing to learn from and — more awkwardly — nothing to score against. The engine can be built without labels, but it cannot be graded without them.

The images are product photography: one item per frame, mostly centred, mostly on a white background. That consistency is doing quiet work in the results, and section 07 has the fix for finding out how much.

---

## 04. Turning Images into Numbers

Two functions do the work, and both are used again unchanged at search time — which matters, because a query described differently from the catalogue would be compared against the wrong thing.

```python
# image pre-processing function

def preprocess_image(filepath):
    
    image = load_img(filepath, target_size = (img_width, img_height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    image = preprocess_input(image)
    
    return image

# featurise image

def featurise_image(image):
    
    feature_vector = model.predict(image)
    
    return feature_vector
```

`preprocess_input` is VGG16's own preprocessing rather than a plain divide-by-255 — it subtracts the channel averages of the ImageNet training set and reorders the colour channels. Using the wrong one here does not throw an error; it quietly shifts every feature vector, and the search degrades without any obvious sign of why.

`np.expand_dims` adds a batch dimension because `predict` expects a stack of images, not one.

```python
# source directory for base images

source_dir = 'data/'

# empty objects to append to

filename_store = []
feature_vector_store = np.empty((0,512))

# pass in & featurise base image set

for image in listdir(source_dir):
    
    print(image)
    
    # append image filename for future lookup
    filename_store.append(source_dir + image)
    
    # preprocess the image
    preprocessed_image = preprocess_image(source_dir + image)
    
    # extract the feature vector
    feature_vector = featurise_image(preprocessed_image)
    
    # append feature vector for similarity calculation
    feature_vector_store = np.append(feature_vector_store, feature_vector, axis = 0)

# save key objects for future use

pickle.dump(filename_store, open('models/filename_store.p', 'wb'))
pickle.dump(feature_vector_store, open('models/feature_vector_store.p', 'wb'))
```

Two stores are built side by side and their order is the only thing linking them. `feature_vector_store` starts as an empty array of width 512 and grows one row per image; `filename_store` grows one filename per image at the same time. The search returns row numbers, and those row numbers are only meaningful because position *n* in one store refers to the same image as position *n* in the other.

Both are pickled so the catalogue is described once rather than on every search. Three hundred images take a couple of minutes on a CPU; the search that follows takes a fraction of a second, and that split is the whole reason the system is usable.

---

## 05. Searching the Catalogue

```python
# load in required objects

model = load_model('models/vgg16_search_engine.h5', compile = False)

filename_store = pickle.load(open('models/filename_store.p', 'rb'))
feature_vector_store = pickle.load(open('models/feature_vector_store.p', 'rb'))

# search parameters

search_results_n = 8
search_image = 'search_image_01.jpg'
        
# preprocess & featurise search image

preprocessed_image = preprocess_image(search_image)
search_feature_vector = featurise_image(preprocessed_image)
        
# instantiate nearest neighbours logic

image_neighbors = NearestNeighbors(n_neighbors = search_results_n, metric = 'cosine')

# apply to our feature vector store

image_neighbors.fit(feature_vector_store)

# return search results for search image (distances & indices)

image_distances, image_indices = image_neighbors.kneighbors(search_feature_vector)
```

`compile = False` skips rebuilding the optimiser and loss function on load. Neither is needed — nothing is being trained — and skipping them avoids a warning about a model saved without them.

`metric = 'cosine'` is the choice that matters most in this block. Cosine distance compares the *direction* of two 512-number vectors and ignores their magnitude, so two photographs of the same shoe under different lighting stay close even when one produces uniformly larger activations than the other. Straight-line distance would treat that brightness difference as a real difference between the products.

`fit` is a misleading name here. `NearestNeighbors` has nothing to learn — the call organises the 300 vectors so that `kneighbors` can search them, and at this size it is effectively storing them. `kneighbors` then returns two aligned arrays: how far away the eight closest images are, and which rows they are.

```python
# get list of filenames for search results

search_result_files = [filename_store[i] for i in image_indices]
```

That single line is where the row numbers become images again, and it is the reason `filename_store` was built alongside the vectors in the first place.

---

## 06. What the Distances Mean

A search returns a distance of 0.113. On its own that number says nothing — the scale has no natural reference point, and cosine distance runs from 0 to 2 in theory while occupying a much narrower band in practice.

A usable reference can be built from the catalogue itself. Every one of the 300 images has a nearest neighbour among the other 299, and the spread of those 300 distances describes how far apart two genuinely similar shoes normally sit in this particular collection.

![The distance from each catalogue image to its own nearest neighbour]({{ "/img/posts/image_search_distances.png" | relative_url }})

The distances run from 0.052 to 0.461, with a median of 0.192. The red heel's best match at 0.113 sits well inside the closer end — nearer to its match than most catalogue images are to theirs. The rain boots' best match at 0.232 sits just above the median, which puts it in ordinary territory rather than in the close range, and matches the observation from section 01 that the second query is a less typical photograph.

This is a reference, not a score. It says where a result sits relative to the rest of the catalogue; it does not say whether the result is correct, because correctness is exactly what an unlabelled catalogue cannot tell you.

---

## 07. Growth & Next Steps

Concrete improvements queued for the next iteration:

- **Label a sample and get a real number.** Tagging even 50 of the 300 images by shoe type would allow precision at 8 to be measured across many queries. Everything in section 01 is currently judged by eye, and a hundred labelled minutes would replace that with a figure.
- **Test the white background.** The catalogue is consistent studio product photography, and the second query — shot on a person — already returned longer distances than the first. Querying with phone photographs against cluttered backgrounds would show whether the engine has learned the shoes or the photographic conditions.
- **Handle a catalogue that grows.** The feature store is rebuilt from scratch every run. Featurising only new images and appending them would make the index maintainable, which is the difference between a demonstration and something that could sit behind a live catalogue.
- **Replace the brute-force search.** Comparing a query against all 300 vectors is instant. Comparing it against 300,000 is not, and approximate nearest neighbour libraries exist precisely for that point.
- **Try a newer backbone.** VGG16 is a 2014 architecture. ResNet or EfficientNet features would be a direct swap in this pipeline and would show how much of the result depends on the network rather than the approach.

---
