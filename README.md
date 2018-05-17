## Writeup On Vehicle Detection Project
### This describes the method used to detect vehicle object in video camera image

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/color_histograms.png
[image4]: ./examples/spatial_bins.png
[image5]: ./examples/heatmap.png
[image6]: ./examples/bounding_boxes.png
[image7]: ./examples/sliding_windows_many.png
[image8]: ./examples/sliding_window.png
[video1]: ./proj_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines /#7 through /#26 of the file called `class_functions.py` (in get_hot_features function).  This was taken from the class material. The function takes in an image and returns hog features.

In the Vehicle-Detection IPython notebook, I am using the get_hog_features function in the extract_features wrapper function in cell /#5 of the IPython notebook in './Vehicle-Detection.ipynb'. In this function the hog features get appended togheter with the spatial bins and the color histogram features in a large feature vector.

In my implementation, I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of some random members of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, which are the parameters I chose for my final implementation:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and looked at the prediction efficiency (The time it took to make a prediction) and tried to make it as close as possible to realtime. With my current computer the parameters that worked best were `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, using the `YUV` color space I got an execution time of 0.9 seconds, running 8 different levels of subwindows accross 3 different scales (see sliding windows section for more details on this specific part). 0.9 seconds is really not that fast for real-time requirements. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG features as well as color features and spatial bins. With HOG only I got to 0.96 accuracy, but when I added color features this went up to 0.98, which is why I decided to add the color histograms and spatial features. Training time is not a requirement here, just prediction rapidity, so we're not trying to optimize the training time. 

The following figure shows the resutlts for color features and spatial bins on a random test image from the training set :

![alt text][image3]

![alt text][image4]

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Based on the reasonoing that closer to the horizon cars appear smaller, I decided do run various experiments with smaller scales around 400px (close to horizon) and larger scale as I defined higher velues for ystart and ystop horizontal bondariers (corresponding to positions LOWER in the images). I settled with the following 8 sets defined in cell /#8 of the .ipynb notebook: 

```python
params= [{'ystart':400,'ystop':465, 'scale':1.0},
 {'ystart':415,'ystop':480, 'scale':1.0},
{'ystart':430,'ystop':495, 'scale':1.5},
{'ystart':400,'ystop':530, 'scale':1.5},
{'ystart':430,'ystop':560, 'scale':2.0},
{'ystart':400,'ystop':595, 'scale':3.5},
{'ystart':465,'ystop':660, 'scale':3.5}]
```

My current scale, pix_per_cell and cell_per_block settings makes it such that there is a 0.5 overlap between the windows of a same set, which yielded satisfactory results, so I chose to call it good and move on. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images showing slinding windows on different test images :

![alt text][image7]

At the beginning I was getting unstable bounding boxes. It was not satisfactory. So I quickly implemented a class to keep track of my rectangle and build heat maps over a set of 15 frames instead of using a single frame. It worked a lot better (see result video)
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./proj_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  This was efficient but the result was very unstable so I decided to accumulate heat over a series of frame. After some trials and errors I figured 8 frames gave me a satisfactory result. 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid one of the test images:

![alt text][image8]

![alt text][image5]

![alt text][image6]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. I used features extractions techniques toghether with a SVC classifier used over a series of sliding windows of different sizes bounded by 8 different ROI to detect vehicle in a video. At the begining I used only histograms of gradients to extract the features but combining this to spatial bins and histogram of colors feature extractions made my SVC more accurate by approximately 2%. The resulting fitted SVC had an accuracty over 98%. It displayed some false positives when there were angles in a region of the image, like a fence bounding the road. Calculating heatmaps and thresholding greatly reduced the artefacts due to false positives but also reduced the rate of detection. To overcome this, heatmaps had to be accumulated over a series of video frames rather than constructed on one single frame. This somewhat solved the problem although regions of images rich in angles and contrast will likely trigger false detections. 

