# Annotation for ImageNet labels for Machine Learning output.
Coding exercise to extend to create an annotation tool for ImageNet labels on Machine Learning output, for use in a job interview or similar context. 
  
The code runs, but _IS DELIBERATELY A BAD IMPLEMENTATION OF AN ANNOTATION TOOL_ 

The task is to fix this code.

# Context 

A company has approached you that wants to classify a large set of sports images according to the ImageNet set of labels. 

They have run their images through a TensorFlow Machine Learning model that predicts the ImageNet label for each image. 

However, there are several problems:
 1. They don't necessarily trust the predictions from the Machine Learning model, so they would like a human to review the output and confirm/reject/fix the labels. 
 1. They have limited budget, so they would like to label them as efficiently as possible.
 1. Their collection of images may include some that are not related to sports. They want to make sure the sports images are filtered out, but they don't otherwise care about these images.

This is a real-world situation that occurs regularly. For this exercise, we will use the open set of sports images from CrowdFlower's Data for Everyone program:

 https://www.crowdflower.com/data-for-everyone/ 

## About ImageNet 

ImageNet uses a classification scheme based on WordNet, where words are grouped by synonyms, called 'synsets'. Each 'synset' is a group of closely related words. These synsets are the labels for this task, which you will see when you run the code. For example, the label for 'racing car' is `'racer, race car, racing car'`.

The predictions from the classifier will therefore look something like this:

`['candle, taper, wax light', 0.079653569], ['wreck', 0.055132806], ['tow truck, tow car, wrecker', 0.038218945], ...`

This indicates that the image being classified has a 0.07965 probability of being a `'candle, taper, wax light'`, a 0.05513 probability of being a `'wreck'`, etc. 

While the classifier is flat, WordNet itself organizes the synsets in hierarchies. For example 'baseball' and 'cricket' could be types of 'sport', and in turn 'sport' could be a type of 'activity'. Generally, items that are closer in the hierarchy tend to be closer in real-life. For example, 'sports car' and 'racing car' are both types of 'cars' in WordNet/ImageNet, and are also closely related in real-life. By contrast, 'sports car' and 'pine tree' are not closely related in WordNet/ImageNet, or in real-life.

 
## Getting started 

A (250MB) subset of the images is available at:

http://www.robertmunro.com/research/test_images.tar 

The starter code is in the same directory as this readme:

`annotate_photos.html`

The output from the classifier is also in this directory at: 

`predictions.js`

The output from the classifier looks like this:

`[['030ae82b1f6feacc14d431248a627dc.jpg', 'packet', 0.66888225, [['packet', 0.66888225], ['matchstick', 0.098503083], ['carton', 0.067100592], ['pencil box, pencil case', 0.041079309], ['book jacket, dust cover, dust jacket, dust wrapper', 0.023718819]]],['03cd8b3b89027a1867bb05dc112b2c.jpg', 'vulture', 0.19094709, [['vulture', 0.19094709], ['wing', 0.1111501], ['warplane, military plane', 0.094249904], ['kite', 0.041520867], ['bald eagle, American eagle, Haliaeetus leucocephalus', 0.033858865]]],['05810e4e3535556aa3a7ea92b374e0.jpg', 'candle, taper, wax light', 0.079653569, [['candle, taper, wax light', 0.079653569], ['container ship, containership, container vessel', 0.057309043], ['wreck', 0.055132806], ['tow truck, tow car, wrecker', 0.038218945], ['spotlight, spot', 0.034531102]]],['06b4793e723e0ade7a5da13117e4b.jpg', 'organ, pipe organ', 0.13145885, [['organ, pipe organ', 0.13145885], ['wall clock', 0.11327242], ['nail', 0.096524686], ['screw', 0.055117268], ['face powder', 0.045105647]]]];`

For each array element, it contains the following:

(0) Name of the file. In the first entry above: '030ae82b1f6feacc14d431248a627dc.jpg'

(1) The most confident label.  In the first entry above: 'packet'

(2) The confidence of the label.  So, in the first entry above, the Machine Learning model was '0.66888225' confident that the image contains a 'packet'.

(3) A list of the top 5 predictions for the image, as per the example above. So, in the first entry above, the Machine Learning algorithm was 66.9% confident that the image contained a packet, 9.9% confident that it contained a matchstick, 6.7% confident that the image contained a carton, etc. 


# Exercise 
 
Your exercise is to improve the code so that it allows a person to quickly and accurately annotate the images with the correct label. 

You can keep the same input/output strategy that is already in the code. In the real-world, we would likely access the data and submit the annotation results via an API. But for this exercise, it is ok to keep using the external `.js` file for input, and for the annotations to be appended to the bottom of page.

The solution should be in Javascript, but it is up to you how you structure the code and what framewords you want to use (jquery, react, angular, etc).

# Potential Solutions
 
There are many extensions to this code, from a 30 minute exercise to make the UI cleaner, to a multiple week exercise that could included quality control and integration with the TensorFlow model to retraining with the new annotations.

## 2 Hour Exercise

For the 2 Hour Exercise, there is a coding and a written component. 

It is recommended that you take 30 minutes to become familiar with the problem and decide on your approach, 60 minutes for the coding exercise, and 30 minutes on the writing exercise.

You may use any resources that are available to you on your machine or on the internet. You can ask the instructor for any clarifications questions, but please complete this as a solo exercise without live input from other people.

### Coding Exercise

Reimplement the code so that it has a better strategy to quickly annotate images. Your strategy can present the images in any order, individually or multiple at a time. You can use any type of button, image selection, or other strategy to make the annotations. You should consider the following general principles in your design, which will sometimes be competing factors:
 1. It is faster to provide a yes/no response than to select from a large list of labels.
 1. It is faster to have the same task repeatedly. For example, it is faster to ask someone "is this basketball" 200 times in a row, rather than mixing up 200 different questions about different sports.
 1. If people are selecting the same response over and over again, they are more likely to stop paying full attention and make errors.

For this 2-hour exercise, you do _not_ need to worry about giving the same task to multiple people to ensure agreeement, or other strategies to detect errors. You only need to worry about the UX and Human-Computer Interaction (HCI) components of a single annotator interacting with the AI output. 

Aim to have working code. It's likely that you won't have time to implement everything that you would like, so the writing exercise allows you to talk about the other strategies that you thought about:

### Writing Exercise

For the writing exercise, pretend like you are addressing the customer about what you have implemented and what you are proposing to build. You can keep the style casual: assume it's a professional email sent to their technical team, not a formal proposal for their executives.

First, write a 1-paragraph description of what you implemented in the coding exercise, justifying each decision. There are many possible solutions that can be implemented in about 60 minutes, so this is more about your reasoning than your exact strategy.

Second, please write a few paragraphs or bullet points proposing other annotation strategies that you might with the customer, covering: 
 1. What additional user interfaces strategies could you build to annotate more labels efficiently and accurately? 
 1. If you used the new annotations to retrain the TensorFlow model, what might be the pros and cons of retraining TensorFlow on the newly labeled images very frequently vs training only a few times or even just once?
 1. How would you evaluate the effectiveness of your strategies to see what worked best?

### Submitting the 2 Hour Exercise

Please email the updated code and written exercise to the instructor when you are complete.
 

# Installation and code source

A (250MB) subset of the images is available at:

http://www.robertmunro.com/research/test_images.tar 

The company made starter code that should run in your browser, in this directory at:

`annotate_photos.html`

The output from the classifier is in:

`predictions.js`
 
For more background about the TensorFlow model and for more context on this problem, see the tutorial at: 

 https://www.tensorflow.org/tutorials/image_recognition 

This tutorial is not required reading for this exercise, but will give you more context if you are not familiar with TensorFlow or ImageNet.

