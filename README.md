# ML_jigsaw-solver_assignment-3

Computer vision can be used to automatically solve jigsaw puzzles by detecting, 
extracting, and analyzing the puzzle pieces' shape, color, texture, and unique 
features. Afterward, the pieces can be assembled using algorithms such as graph-based 
matching, template matching, or deep learning-based methods. This requires a combination
of image processing, pattern recognition, and machine learning algorithms. Automating jigsaw puzzle-solving has many potential applications, including robotics, gaming, and image and video processing.

Permutation Invariance:
A function is a permutation invariant if its output does not change by changing the ordering of its input. 
A jigsaw puzzle is also permutation invariance. No matter what the ordering of puzzle pieces are the output would always be fixed.

Overview of the code:
Here we first took a dataset of normal images to create a dataset of shuffled 2x2 images. We conducted this data generation by diving the images into 2x2 grid format and then shuffle it randomly to store in the new dataset. This dataset was then divided for training and testing datasets.
We then trained a model to rearrange this shuffled image to the original form while storing the position of the image pices in an array format.
We then tested this model on the test set accessed by the label number of the image.

Inputs:

![image](https://user-images.githubusercontent.com/76091761/234572368-361a59f0-eb1e-42d3-8056-4177257018a0.png) 

![image](https://user-images.githubusercontent.com/76091761/234572469-7d6f41fc-4ee6-4204-bd70-211db9200f85.png)


Outputs:

![image](https://user-images.githubusercontent.com/76091761/234572537-0e860f15-6d72-4498-898d-00152e39cdd9.png)


![image](https://user-images.githubusercontent.com/76091761/234572504-959418a3-3bce-4721-9a3c-ce55635683c8.png)




