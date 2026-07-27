---
layout: post
title: Classifying Fruit Photos with a Convolutional Neural Network
image: "/img/posts/cnn_fruit_classification.svg"
tags: [Deep Learning, Computer Vision, Python]
summary: "A small convolutional network learns six fruits from 360 photographs. Four standard improvements are added one at a time — dropout, augmentation, a pre-trained network and an architecture search — each scored on the same photographs."
stack: "Python · TensorFlow · Keras"
metrics:
  - value: "98%"
    label: "accuracy score"
  - value: "360"
    label: "training photos"
  - value: "+13pp"
    label: "gained over the baseline"
---

Six fruits, sixty photographs of each, and a network that has to tell them apart. The first one I trained learned its photographs perfectly and then got roughly one in six wrong on photographs it had not seen — it had learned the pictures rather than the fruit. This project starts from that baseline and adds four standard fixes one at a time — dropout, image augmentation, a pre-trained network, and an automated architecture search — measuring each one on the same photographs so the improvements can be read against each other rather than described in the abstract.

---

# Table of Contents

- [00. Project Overview](#00-project-overview)
- [01. Results](#01-results)
- [02. Model Overview](#02-model-overview)
- [03. Data Overview](#03-data-overview)
- [04. Data Preparation](#04-data-preparation)
- [05. The Baseline Network](#05-the-baseline-network)
  - [Architecture](#bl-architecture)
  - [Compiling & Training](#bl-training)
  - [Where the Parameters Live](#bl-parameters)
- [06. Four Improvements](#06-four-improvements)
  - [Dropout](#imp-dropout)
  - [Image Augmentation](#imp-augmentation)
  - [Transfer Learning](#imp-transfer)
  - [Architecture Search](#imp-tuner)
- [07. Comparing Every Version](#07-comparing-every-version)
- [08. Growth & Next Steps](#08-growth--next-steps)

---

## 00. Project Overview

**Context**

Image classification is the standard first problem in deep learning, and this version of it is deliberately small: six classes, sixty training photographs each. Four techniques sit on top of that baseline — dropout, augmentation, transfer learning and an automated architecture search — and each is added on its own and measured before the next one goes in. Every version changes exactly one thing, so every number in the results table can be traced back to a single decision.

**Actions**

Five versions of the same problem, trained in sequence. The first is a plain convolutional network — two convolutional blocks and a single dense layer — which sets the baseline. The next four each change one thing: dropout, image augmentation, a pre-trained VGG16 in place of the learned features, and an architecture chosen by Keras Tuner rather than by hand. All five saved checkpoints were then loaded back and scored on the same three sets of photographs — the ones they trained on, the ones used to pick the best epoch, and sixty they had never touched.

**Applications**

The same setup transfers to any which-one-is-it question asked about a photograph. The nearest version of it is sorting on a production line — grading produce, or separating good units from defective ones — where the classes are still a handful of folders of example images and only the subject changes.

**Growth & Next Steps**

The honest limitation is the size of the test set. Sixty photographs means a single image is worth 1.7 percentage points, so small differences between versions cannot be separated from luck. The natural next step is a larger and harder set of held-out photographs — ideally shot in ordinary kitchen lighting rather than drawn from the same clean source as the training images.

---

## 01. Results

Each version was scored on all three sets of photographs. The training column shows what the network scores on the photos it learned from; the other two show photographs it never trained on.

| Version | Training (360) | Validation (180) | Test (60) | Parameters |
|---|---|---|---|---|
| Baseline | 100.0% | 83.9% | 85.0% | 1,058,950 |
| Dropout | 99.7% | 87.8% | 81.7% | 1,058,950 |
| Augmentation | 96.1% | 97.2% | 91.7% | 1,058,950 |
| Transfer (VGG16) | 99.2% | 97.2% | 95.0% | 15,780,678 |
| Tuner | 97.8% | 99.4% | 98.3% | 2,717,542 |

![Accuracy on photos the network trained on versus photos it had never seen]({{ "/img/posts/cnn_fruit_generalisation.png" | relative_url }})

**The baseline scored perfectly on its own photographs but worse on new ones.** Every one of the 360 training images was classified correctly, which sounds like success but it is not. A 16 point drop on unseen photographs means overfitting - the network memorised the 360 specific photographs in front of it rather than learn what separates one fruit from another.

**Changing the photographs helped more than making the network independent.** Augmentation — randomly rotating, shifting, zooming and flipping each training image — moved validation accuracy from 83.9% to 97.2% without altering a single layer. It also closed the gap between seen and unseen photographs entirely.

**The 15-million-parameter pre-trained network did not win.** VGG16 brought features learned from millions of images and landed at 95.0% on the test set, ahead of augmentation alone but behind the smaller network the tuner arrived at, which reached 98.3% with a sixth of the parameters.

**Dropout is the one version with no clear verdict.** It improved validation accuracy by 3.9 points but scored two photographs worse on the test set, so the two measurements disagree. With only sixty test photographs, a two-image swing imay be due to ordinary luck, which leaves the question of low score open.

---

## 02. Model Overview

A photograph, to a computer, is a grid of numbers: a 128×128 colour image is 49,152 values between 0 and 255. The approach of wiring every pixel to a decision layer fails, because a pixel's meaning comes from the pixels around it rather than from its position in the grid — a banana in the top-left corner and the same banana in the bottom-right are the same banana, and to a position-by-position model they look unrelated.

A **convolutional neural network** solves this by learning small patterns and sliding them across the whole image. Each filter is a small stencil — in our case three pixels by three — that is compared against every position in turn, producing a map of where that pattern appears. The stencils begin as random numbers and training adjusts them until one has become an edge detector, another responds to a particular curve, another to a colour transition. Stacking these layers, with a shrinking step in between, lets later layers combine simple patterns into complicated ones. The final layers flatten the result into a list and turn it into six confidence scores, one per fruit.

---

## 03. Data Overview

The dataset is a folder of fruit photographs split three ways, with one subfolder per class. The folder name *is* the label — there is no separate labels file — and the split is even, so no class is favoured.

| Set | Photos per class | Total | Purpose |
|---|---|---|---|
| Training | 60 | 360 | What the network learns from |
| Validation | 30 | 180 | Checked after every epoch to save the best version |
| Test | 10 | 60 | Never seen during training; the final score |

The six classes are apple, avocado, banana, kiwi, lemon and orange. 

---

## 04. Data Preparation

Preparation is handled by Keras' image generators rather than by hand, and it does three things: resize every photograph to 128×128 so the network's input is a fixed shape, divide every pixel by 255 to normalise values between a 0–1 range, and read the images in batches of 32 so memory holds a handful of photographs at a time instead of all 360.

```python
# data flow parameters

training_data_dir = 'data/training'
validation_data_dir = 'data/validation'
batch_size = 32
img_width = 128
img_height = 128
num_channels = 3
num_classes = 6

# image generators

training_generator = ImageDataGenerator(rescale = 1./255)
validation_generator = ImageDataGenerator(rescale = 1./255)

# image flows

training_set = training_generator.flow_from_directory(directory = training_data_dir,
                                                      target_size = (img_width,img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')

validation_set = validation_generator.flow_from_directory(directory = validation_data_dir,
                                                          target_size = (img_width,img_height),
                                                          batch_size = batch_size,
                                                          class_mode = 'categorical')
```

The parameters at the top are referred to by name for the rest of the project: `img_width` and `img_height` fix every photograph at 128×128, `num_channels = 3` is the red, green and blue layers of a colour image, and `num_classes = 6` is the six fruits. `class_mode = 'categorical'` returns each label as six values with a 1 in the correct position rather than as a single number, which is the shape the output layer and the loss function expect.

Rescaling is required as the network learns by nudging its internal numbers in small steps, and inputs in the hundreds make those steps overshoot instead of settle — the same reason we had features normalised before a regression. Further, whatever preprocessing happens here has to happen identically at prediction time, or the model produces confident nonsense.

`flow_from_directory` sorts the class folders alphabetically, so apple becomes 0 and orange becomes 5. Every script then hard-codes `labels_list` in that same alphabetical order to turn a prediction back into a word — which works only because the two orders agree. Rename a folder and the predictions silently become wrong.

---

## 05. The Baseline Network

### Architecture {#bl-architecture}

Two convolutional blocks, then a small decision layer:

```python
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', input_shape = (img_width,img_height,num_channels)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
```

Each block does the same three things. `Conv2D` slides 32 different 3×3 stencils across the image and records where each one matches, with `padding = 'same'` keeping the output the same width and height so border pixels are not under-sampled. `Activation('relu')` sets every negative result to zero, which is what allows the network to represent something other than a straight line. `MaxPooling2D` then halves the width and height by keeping only the strongest response in each 2×2 square — the image shrinks, the pattern survives, and the next layer sees a larger area for the same cost.

`Flatten` turns the final stack of feature maps into one long list, `Dense(32)` weighs everything against everything, and the last `Dense(6)` with `softmax` converts the result into six probabilities that sum to one.

### Compiling & Training {#bl-training}

```python
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

save_best_model = ModelCheckpoint(filepath = model_filename,
                                  monitor = 'val_accuracy',
                                  mode = 'max',
                                  verbose = 1,
                                  save_best_only = True)

history = model.fit(x = training_set,
                    validation_data = validation_set,
                    batch_size = batch_size,
                    epochs = num_epochs,
                    callbacks = [save_best_model])
```

`categorical_crossentropy` is the loss for a one-of-several choice: it punishes confident wrong answers far more than hesitant ones, which is what pushes the network towards being both right and sure. `adam` is the optimiser that decides how big each correction is. Accuracy is reported but plays no part in learning — the network optimises loss, and accuracy is the human-readable scoreboard.

The checkpoint matters more than it looks. Networks do not improve indefinitely; they peak and then decline as they start memorising. `save_best_only = True` watching `val_accuracy` means the file on disk is overwritten only when performance on unseen photographs reaches a new high, so 50 epochs of training leave behind the best epoch rather than the last one.

### Where the Parameters Live {#bl-parameters}

`model.summary()` on this network is the most persuasive argument for the whole convolutional design:

| Layer group | Parameters | Share |
|---|---|---|
| Two convolutional layers | 10,144 | 1.0% |
| `Dense(32)` after `Flatten` | 1,048,608 | 99.0% |
| Output layer | 198 | — |

The convolutions — the part actually doing the seeing — cost one percent of the network. The single fully-connected layer behind them costs the other ninety-nine, because flattening a 32×32×32 stack produces 32,768 values and connecting each to 32 neurons takes a million weights. Convolution is astonishingly cheap; connecting everything to everything is astonishingly expensive.

---

## 06. Four Improvements

Everything below reuses the machinery already described — same generators, same compile step, same checkpointing, same scoring. Only the stated change differs.

### Dropout {#imp-dropout}

The baseline scored 100% on its training photographs. That is memorisation, and dropout is the standard answer: during training, half the neurons in a layer are switched off at random on every pass, so no single neuron can become the one that recognises a particular photograph.

```python
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
```

One line, placed after the dense layer — which is where over-fitting concentrates, since that is where 99% of the parameters live. Training accuracy fell from 100% to 99.7% and validation accuracy rose from 83.9% to 87.8%, closing the gap between them from 16 points to 12. The test set moved the other way, from 85.0% to 81.7%, and the honest reading is that two photographs out of sixty is not a result. What is visible in the confusion counts is that the damage was concentrated: this version put five of the ten bananas in the lemon column.

### Image Augmentation {#imp-augmentation}

Dropout attacks memorisation by changing the network. Augmentation attacks the underlying cause instead — 360 photographs is not much to learn from — by inventing more of them.

```python
training_generator = ImageDataGenerator(rescale = 1./255,
                                        rotation_range = 20,
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2,
                                        zoom_range = 0.1,
                                        horizontal_flip = True,
                                        brightness_range = (0.5,1.5),
                                        fill_mode = 'nearest')

validation_generator = ImageDataGenerator(rescale = 1./255)
```

Every time a photograph is drawn for training it arrives slightly altered — rotated up to 20 degrees, shifted up to a fifth of the frame, zoomed, flipped, brightened or darkened. The network effectively never sees the same image twice, so it cannot memorise the set, and each setting encodes something true about photographs of fruit: they are not always upright, centred, at the same distance, or shot in the same light.

The second line is the one that is easy to get wrong. The validation generator keeps only the rescaling. Augmenting the data you are scoring against would make the score meaningless.

This was the single largest improvement in the project: validation accuracy 83.9% → 97.2%, test accuracy 85.0% → 91.7%, with the network architecture completely untouched. It also inverted the seen/unseen gap — the model now scores slightly *worse* on training photographs (96.1%) than on unseen ones, because during training it only ever sees distorted versions while validation photographs arrive clean.

### Transfer Learning {#imp-transfer}

The next idea is to stop learning what an edge looks like. VGG16 is a network already trained on millions of photographs across a thousand categories, and its convolutional layers have long since learned edges, textures and shapes. Transfer learning keeps those layers, freezes them, and trains only a new decision head on top.

```python
vgg = VGG16(input_shape = (img_width, img_height, num_channels), include_top = False)

for layer in vgg.layers:
    layer.trainable = False

flatten = Flatten()(vgg.output)

dense1 = Dense(128, activation = 'relu')(flatten)
dense2 = Dense(128, activation = 'relu')(dense1)

output = Dense(num_classes, activation = 'softmax')(dense2)

model = Model(inputs = vgg.inputs, outputs = output)
```

`include_top = False` discards VGG16's original thousand-way classifier and keeps the feature extractor. Freezing every layer means the borrowed knowledge is used, not overwritten — of 15,780,678 parameters, only 1,065,990 are trainable. The preprocessing also changes: VGG16 expects the exact channel treatment it was trained with, so `preprocess_input` replaces the `1./255` rescale in both the generators and the prediction function.

It reached 97.2% validation and 95.0% test accuracy in **10 epochs** rather than 50 — the borrowed features mean there is far less to learn. It did not, however, beat the much smaller network in the next section.

### Architecture Search {#imp-tuner}

Every choice so far — 32 filters, two convolutional blocks, 32 neurons in the dense layer — was a guess. Keras Tuner replaces the guessing with a search: the architecture is described as a range of possibilities and the tuner trains candidates to find out which combination works.

```python
def build_model(hp):

    model = Sequential()

    model.add(Conv2D(filters = hp.Int("Input_Conv_Filters", min_value = 32, max_value = 128, step = 32), kernel_size = (3,3), padding = 'same', input_shape = (img_width,img_height,num_channels)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    for i in range(hp.Int("n_Conv_Layers", min_value = 1, max_value = 3, step = 1)):

        model.add(Conv2D(filters = hp.Int(f"Conv_{i}_Filters", min_value = 32, max_value = 128, step = 32), kernel_size = (3,3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D())

    model.add(Flatten())

    for j in range(hp.Int("n_Dense_Layers", min_value = 1, max_value = 4, step = 1)):

        model.add(Dense(hp.Int(f"Dense_{j}_Neurons", min_value = 32, max_value = 128, step = 32)))
        model.add(Activation('relu'))

        if hp.Boolean("Dropout"):

            model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = hp.Choice("Optimizer", values = ['adam','RMSProp']),
                  metrics = ['accuracy'])

    return model
```

Every `hp.` call is a decision handed to the tuner rather than made in advance: how many filters, how many convolutional blocks, how many dense layers and how wide, whether to use dropout at all, and which optimiser.

```python
tuner = RandomSearch(hypermodel = build_model,
                     objective = 'val_accuracy',
                     max_trials = 3,
                     executions_per_trial = 2,
                     directory = os.path.normpath('C:/'),
                     project_name = 'fruit-cnn',
                     overwrite = True)

tuner.search(x = training_set, validation_data = validation_set, epochs = 5, batch_size = batch_size)
```

The objective is `val_accuracy` and not training accuracy — tuning on the latter would simply select the best memoriser. `executions_per_trial = 2` trains each candidate twice and averages, because a single run of a small network is noisy enough to crown the wrong architecture.

The search settled on a deeper, wider network than the hand-built one: three convolutional blocks of 96, 64 and 64 filters, a 160-neuron dense layer, dropout switched on, Adam as the optimiser. Retrained for the full 50 epochs it reached 99.4% validation and 98.3% test accuracy — the best result in the project.

Two caveats belong with that number. `max_trials = 3` against a search space of several hundred combinations is a demonstration of the mechanism, not an exhaustive search. And the architecture it landed on independently re-derived a lesson from earlier in the sequence: it chose dropout, and it chose more convolutional depth.

---

## 07. Comparing Every Version

Training a network and scoring it are separate jobs. Each version saved its best checkpoint during training, and the block below is what reloads that checkpoint and puts it in front of the sixty test photographs. The same block runs at the end of every version, so no model is being judged on a different footing — only the checkpoint filename changes, and for VGG16 the `1./255` rescale is swapped for `preprocess_input`.

```python
# load model

model = load_model(model_filename)

# image pre-processing function

def preprocess_image(filepath):
    
    image = load_img(filepath, target_size = (img_width, img_height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    image = image * (1./255)
    
    return image

# image prediction function

def make_prediction(image):
    
    class_probs = model.predict(image)
    predicted_class = np.argmax(class_probs)
    predicted_label = labels_list[predicted_class]
    predicted_prob = class_probs[0][predicted_class]
    
    return predicted_label, predicted_prob

# loop through test data

source_dir = 'data/test/'
folder_names = ['apple', 'avocado', 'banana', 'kiwi', 'lemon', 'orange']
actual_labels = []
predicted_labels = []
predicted_probabilities = []
filenames = []

for folder in folder_names:
    
    images = listdir(source_dir + '/' + folder)
    
    for image in images:
        
        proccessed_image = preprocess_image(source_dir + '/' + folder + '/' + image)
        predicted_label, predicted_probability = make_prediction(proccessed_image)
        
        actual_labels.append(folder)
        predicted_labels.append(predicted_label)
        predicted_probabilities.append(predicted_probability)
        filenames.append(image)

# create dataframe to analyse

predictions_df = pd.DataFrame({"actual_label": actual_labels,
                              "predicted_label": predicted_labels,
                              "predicted_probability": predicted_probabilities,
                              "filename": filenames})

predictions_df['correct'] = np.where(predictions_df['actual_label'] == predictions_df['predicted_label'], 1, 0)

# overall test set accuracy

test_set_accuracy = predictions_df['correct'].sum() / len(predictions_df)

# confusion matrix

confusion_matrix = pd.crosstab(predictions_df['predicted_label'], predictions_df['actual_label'])
```

`preprocess_image` has to repeat exactly what the training generator did — resize to 128×128, add a batch dimension because `predict` expects a stack of images rather than one, divide by 255. `make_prediction` returns two things: `argmax` picks the highest of the six scores and turns it back into a word via `labels_list`, and the score itself is kept as the model's confidence in that answer. Keeping the probability rather than only the label is what makes the confidence table further down possible.

The loop walks the test folders in the same alphabetical order the training generator used, so predictions and labels stay aligned. Everything lands in a dataframe with a `correct` flag, which reduces accuracy to a mean and leaves the individual predictions intact for anything else worth asking.

Overall accuracy says how often a model is wrong. The confusion matrix says what it is wrong *about*, which is far more useful:

![Test-set confusion matrices for the baseline and the tuned network]({{ "/img/posts/cnn_fruit_confusion.png" | relative_url }})

The baseline's nine errors are spread across five of the six fruits — avocado is the only one it never mislabels. Two bananas are called lemons, and apple, kiwi and orange are confused with one another in both directions. The tuned network has one error left: an apple it calls a kiwi.

There is a second signal worth reading alongside accuracy — how confident each model is when it gets an answer wrong.

| Version | Mean confidence when right | Mean confidence when wrong |
|---|---|---|
| Baseline | 0.94 | 0.72 |
| Dropout | 0.93 | 0.77 |
| Augmentation | 0.96 | 0.52 |
| Transfer (VGG16) | 0.99 | 0.91 |
| Tuner | 0.94 | 0.40 |

The tuned network hesitates when it is wrong — 0.40 on its single mistake, against 0.94 when correct — so a confidence threshold would catch it. VGG16 is the opposite: it is 91% sure of the answers it gets wrong, which makes its errors much harder to filter out automatically. For anything that needs to hand uncertain cases to a human, that difference matters as much as the three points of accuracy between them.

---

## 08. Growth & Next Steps

Concrete improvements queued for the next iteration:

- **A bigger and harder test set.** Sixty photographs cannot separate a two-point difference from luck, and the whole comparison above is limited by it. Photographs shot in ordinary kitchen lighting, against cluttered backgrounds, would also test something the current set does not: whether the network has learned the fruit or the photographic conditions.
- **Fine-tune VGG16 rather than freeze it.** Unfreezing the last convolutional block and training it at a low learning rate would let borrowed features adapt to fruit specifically — the standard next step, and the fairest test of whether transfer learning really was beaten here.
- **Run the tuner properly.** Three trials chose a good architecture; fifty would say whether it is the right one, and Hyperband would get there for a similar amount of compute.
- **Look at what the filters learned.** The feature maps of the first convolutional layer can be rendered as images, which turns the explanation in section 02 into something visible rather than described.

---
