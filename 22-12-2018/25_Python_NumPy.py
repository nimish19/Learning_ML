
"""
Module 0: Introduction

    What is Machine Learning
    Types of ML: Supervised, Unsupervised, Reinforcement
    Types of ML problems: Regression, Classification

Module 1: Linear Models

    Linear Regression
    Logistic Regression

Module 2: Model Evaluation

    Training and Validation
    Model Evaluation Metrics - Accuracy, RMSE, ROC, AUC, Confusion Matrix, Precision, Recall, F1 Score
    Overfitting and Bias-Variance trade-off
    Regularization (L1/L2)
    K-fold Cross Validation

Module 3: Tree-based Models

    Decision Trees
    Bagging and Boosting
    Random Forest
    Gradient Boosting Machines
    Feature Importance

Module 4: Model Selection

    Model Pipelines
    Feature Engineering
    Ensemble Models (Advanced)
    Unbalanced Classes (Advanced)

"""



"""
https://realpython.com/tutorials/data-science/

"Data science" is just about as broad of a term as they come. 
It may be easiest to describe what it is by listing its more concrete components:
    
Data Exploration & Analysis (EDA).

    Included here: Pandas; NumPy; SciPy; a helping hand from Python’s Standard Library.

Data visualization. A pretty self-explanatory name. Taking data and turning it into something colorful.

    Included here: Matplotlib; Seaborn; Datashader; others.

Classical machine learning. 
Conceptually, we could define this as any supervised or unsupervised learning task that is not deep learning (see below). 
Scikit-learn is far-and-away the go-to tool for implementing classification, regression, clustering, and dimensionality reduction, 
while StatsModels is less actively developed but still has a number of useful features.

    Included here: Scikit-Learn, StatsModels.


Deep learning. This is a subset of machine learning that is seeing a renaissance, 
and is commonly implemented with Keras, among other libraries. 
It has seen monumental improvements over the last ~5 years, such as AlexNet in 2012, 
which was the first design to incorporate consecutive convolutional layers.

    Included here: Keras, TensorFlow, and a whole host of others.

Reinforcement Learning
    

Data storage and big data frameworks. 
Big data is best defined as data that is either literally too large to reside on a single machine, 
or can’t be processed in the absence of a distributed environment. 
The Python bindings to Apache technologies play heavily here.

    Apache Spark; Apache Hadoop; HDFS; Dask; h5py/pytables.


Odds and ends. Includes subtopics such as natural language processing, and image manipulation with libraries such as OpenCV.

    Included here: nltk; Spacy; OpenCV/cv2; scikit-image; Cython.

Deployment of Machine Learning Models
    AWS
    Dockers
    API
    Web ( HTML, CSS and JS )


"""


"""
Introduce the concept of Scalar, Vectors and Tensors 
(Scalar_Vector_Tensor)

Scalar = When we want to store 1 piece of information ( value ) for a given physical quantity.
E.g. = Temperature and Pressure

Scalar has one component
Scalar is tensor of rank zero

Vector = When we want to store 2 piece of information ( value and direction ) for a given physical quantity.
E.g. = Position, Force and Velocity

v = v1*x + v2*y + v3z
[v1 v2 v3 ] as an array 

or 
as an columnar
[ v1 ]
[ v2 ]
[ v3 ]

Vector has three component
Vector is a Tensor of rank one, one basis vector per component

Tensor = When we want to store 3 piece of information ( value and direction and plane ) for a given physical quantity.
E.g. = stress , forces in side an object    
[v11  v12  v13 ]
[v21  v22  v23 ]
[v31  v32  v33 ]

Tensor has nine component
Tensor is tensor of rank two

If the rank is three then the components are 27



Introdcue the visualisation of 1D, 2D and 3D using an image and define axis
(1D_2D_3D_Visualisation.png)
(3d_visualisation.png)


Color-image data for single image is typically stored in three dimensions. 
Each image is a three-dimensional array of (height, width, channels), 
where the channels are usually red, green, and blue (RGB) values. 
One 256x256 RGB images would have shape (256, 256, 3). 
(An extended representation is RGBA, where the A–alpha–denotes the level of opacity.)
One 256x256 ARGB images would have shape (256, 256, 4). 



Color-image data for multiple images is typically stored in four dimensions. 
A collection of images is then just (image_number, height, width, channels). 
One thousand 256x256 RGB images would have shape (1000, 256, 256, 3). 


(An extended representation is RGBA, where the A–alpha–denotes the level of opacity.)

"""

"""
Difference between LIST and ARRAY 

Arrays support vectorised operations, while lists don’t.
Once an array is created, you cannot change its size. 
You will have to create a new array or overwrite the existing one.
Every array has one and only one dtype. All items in it should be of that dtype.
An equivalent numpy array occupies much less space than a python list of lists.



Arrays are the main data structure used in machine learning.

In Python, arrays from the NumPy library, called N-dimensional arrays or the ndarray, 
are used as the primary data structure for representing data.



When it comes to computation, there are really three concepts that lend NumPy its power:

Vectorization
Vectorization is a powerful ability within NumPy to express operations as occurring on entire arrays rather than their individual elements
This practice of replacing explicit loops with array expressions is commonly referred to as vectorization. 
When looping over an array or any data structure in Python, there’s a lot of overhead involved. 
Vectorized operations in NumPy delegate the looping internally to highly optimized C and Fortran functions, 
making for cleaner and faster Python code.
    
    


Broadcasting
The term broadcasting describes how NumPy treats arrays with different shapes during arithmetic operations. 
Subject to certain constraints, the smaller array is "broadcast" across the larger array so that they have compatible shapes. 
Broadcasting provides a means of vectorizing array operations so that looping occurs in C instead of Python. 



Indexing


"""


# NumPy == Numerical Python
# library consisting of multidimensional array objects 
# and a collection of routines for processing those arrays. 

# Mathematical and logical operations on arrays.
# Fourier transforms and routines for shape manipulation
# NumPy has in-built functions for linear algebra and random number generation.


# NumPy – A Replacement for MatLab
# NumPy is often used along with packages like SciPy (Scientific Python) and 
# Mat−plotlib (plotting library).

# pip install numpy 


# N-dimensional array type called ndarray.
# It describes the collection of items of the same type. 
# Items in the collection can be accessed using a zero-based index.
# Any item extracted from ndarray object (by slicing)
# is represented by a Python object of one of array scalar types.

# https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
# https://www.tutorialspoint.com/numpy/numpy_array_manipulation.htm

# Three important concepts
# 1. List to Array
# 2. Array Indexing
# 3. Array Slicing
# 4. Array Reshaping


import numpy as np

"""
There are a couple of mechanisms for creating arrays in NumPy:
 a. Conversion from other Python structures (e.g., lists, tuples).
 b. Built-in NumPy array creation (e.g., arange, ones, zeros, etc.).
 c. Reading arrays from disk, either from standard or custom formats 
     (e.g. reading in from a CSV file).
"""


# Convert your list data to NumPy arrays

# One Dimensional Array ( 1D Array ) 
a = [1,2,3,4,5,6,7,8,9]
print (type(a))
print (a)  
# it always prints the values with comma seperated , thats list


# Converting list to Array 
x = np.array( a ) 
print (type(x))

print (x)
# it always prints the values WITHOUT comma seperated , thats ndarray

# to print the dimension of the array 
print (x.ndim)

# to print the shape of the array 
print (x.shape)

# to print the data type of the elements of array 
print (x.dtype)
# For a 1D array, the shape would be (n,) where n is the number of elements in your array.



# Array Indexing will always return the data type object 
print (x[0])
print (x[2])
print (x[-1])



"""
x[start:end] # items start through the end (but the end is not included!)
x[start:]    # items start through the rest of the array
X[:end]      # items from the beginning through the end (but the end is not included!)
"""

# Array Slicing will always return ndarray  
# data [ from : to ]
print (x[:])  # blank means from start or till end
print (x[0:]) 
print (x[:3]) 
print (x[0:2])



"""
Reshaping is changing the arrangement of items so that shape of the array changes while maintaining the same number of dimensions.

Flattening, however, will convert a multi-dimensional array to a flat 1d array. And not any other shape.
"""

# Reshaping to 3 Rows and 3 Columns
b = x.reshape(3,3)
print (x)
print (b.ndim)
print (b.shape)
print (b.dtype)


# Reshaping to 3 layers of 3 Rows and 3 Columns 
x = np.array( [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27] )
x = x.reshape(3,3,3)
 
print (x)
print (x.ndim)
print (x.shape)
print (x.dtype)

       


"""
For 1D array, shape return a  tuple with only 1 component (i.e. (n,))
For 2D array, shape return a  tuple with only 2 components (i.e. (n,m))
For 3D array, shape return a  tuple with only 3 components (i.e. (n,m,k) )
"""


# Creating 2 Dimensional Array 
x = np.array( [ [1, 2, 3], [4, 5, 6] ] )
print (type(x))
print (x)
print (x.ndim)
print (x.shape)
print (x.dtype)
# For 2D array, return a shape tuple with only 2 elements (i.e. (n,m))



# Array Indexing 
print (x)

print (x[0])
print (type(x[0]))
print (x[0,0])
print (x[0,1])
print (x[0,2])


print (x[1])
print (type(x[1]))
print (x[1,0])
print (x[1,1])
print (x[1,2])



# Creating 3 Dimensional Array
x = np.array([ [ [1, 2, 3], [4, 5, 6] ], [ [11, 22, 33], [44, 55, 66] ], [ [111, 222, 333], [444, 555, 666] ]  ] )
print (x)
print (x.ndim)
print (x.shape)
print (x.dtype)
# For 3D array, return a shape tuple with only 3 elements (i.e. (n,m,k) )
# We introduced the concept of another layer represented by n


# Array Indexing 
print (x)
print (x[0])
print (x[0,0])
print (x[0,0,0])




# Creating multi dimensional array 

# One Dimensional Array 
x = np.array( [1,2,3] )  # ndmin = 1
print (type(x))
print (x)
print (x.ndim )
print (x.shape )
print (x.dtype )
print (x[0])
# If we access on the zeroth location we would get 1

# Two Dimensional Array with only one row of data
x = np.array( [1,2,3] , ndmin = 2) 
print (type(x))
print (x)
print (x.ndim)
print (x.shape)
print (x.dtype)
print (x[0])
# If we access on the zeroth location we would get the 1D array [1 2 3]



# Three Dimensional Array with only one row of data 
x = np.array( [1,2,3] , ndmin = 3) 
print (type(x))
print (x)
print (x.ndim)
print (x.shape)
print (x.dtype)
print (x[0])
# If we access on the zeroth location we would get 2D array [[1 2 3]]

"""
Array         Dimen     Shape
[1 2 3]         1       (3,)

[[1 2 3]]       2       (1,3)

[[[1 2 3]]]     3       (3,)

"""


#-----------------------------------------------------------------------------
# Numpy supports all data types likes bool, integer, float, complex etc.
# They are defined by the numpy.dtype class 

import numpy as np 

x = np.float32(1.0) 
print (x) 
print (type(x)) 

x = np.float64(1.0)
print (x) 
print (type(x)) 

a
x = np.int_([1,2,4]) 
print ( x )
print (type(x)) 




x = np.array([1, 2, 3], dtype = complex) 
print (x)
print (x.ndim)
print (x.shape)
print (x.dtype)







"""
There are a couple of mechanisms for creating arrays in NumPy:
 a. Conversion from other Python structures (e.g., lists, tuples).
 b. Built-in NumPy array creation (e.g., arange, ones, zeros, etc.).
 c. Reading arrays from disk, either from standard or custom formats 
     (e.g. reading in from a CSV file).
"""
# Using the built in function arange 

# Arange function will generate array from 0 to size-1 
# arange is similar to range function but generates an array , where in range gives you a list

x = np.arange(20, dtype=np.uint8)
print (x)
print (x.ndim)
print (x.shape)
print (x.dtype)



# zeros(shape) -- creates an array filled with 0 values with the specified 
#                  shape. The default dtype is float64.

x = np.zeros((3, ))
print (x)
print (x.ndim)
print (x.shape)
print (x.dtype)


x = np.zeros((3, 3))
print (x)
print (x.ndim)
print (x.shape)
print (x.dtype)


x = np.zeros((3, 3, 3))
print (x)
print (x.ndim)
print (x.shape)
print (x.dtype)




# ones(shape) -- creates an array filled with 1 values. 


x = np.ones((3, ), dtype=np.int8 )
print (x)
print (x.ndim)
print (x.shape)
print (x.dtype)


x = np.ones((3, 3), dtype=np.int8 )
print (x)
print (x.ndim)
print (x.shape)
print (x.dtype)


x = np.ones((3, 3, 3), dtype=np.int8 )
print (x)
print (x.ndim)
print (x.shape)
print (x.dtype)


#-----------------------------------------------------------------------
# linspace() -- creates arrays with a specified number of elements, 
# and spaced equally between the specified beginning and end values.

x = np.linspace(1, 4, 10, dtype = np.float64)
print (x)
print (x.ndim)
print (x.shape)
print (x.dtype)


#np.around(x,2)



#random.random(shape) – creates arrays with random floats over the interval [0,1].
x = np.random.random((2,3))*1000
print (x)
print (x.ndim)
print (x.shape)
print (x.dtype)


# np.identity() to create a square 2d array with 1's across the diagonal
print (np.identity(n = 5))      # Size of the array



# np.eye() to create a 2d array with 1's across a specified diagonal
np.eye(N = 3,  # Number of rows
       M = 5,  # Number of columns
       k = 1)  # Index of the diagonal (main diagonal (0) is default)




# NaN can be defined using the following constant
print (np.nan)
print(type(np.nan))
# Infinite value can be expressed using the following contant 
print (np.inf)
print(type(np.inf))


x = np.array( [1,2,3], dtype=np.float ) 
print (x)
print(x.dtype)


x[0] = np.nan
x[2] = np.inf
print (x)



"""
# Replace nan and inf with -1. Don't use arr2 == np.nan
missing_bool = np.isnan(arr2) | np.isinf(arr2)
arr2[missing_bool] = -1  
"""


x = np.array( [1,2,3] , ndmin = 1) 
print (type(x))
print (x)
print (x.ndim)
print (x.shape)
print (x.dtype)


print("Sum is: ", x.sum())
print("Average/Mean value is: ", x.mean())
print("Max value is: ", x.max())
print("Min value is: ", x.min())


#print("Median value is: ", np.median(x))
#print("Correlation coefficient value is: ", np.corrcoef(x))
#print("Standard Deviation is: ", np.std(x))



x = np.array( [[1,2,3],[4,5,6]]) 
print (type(x))
print (x)
print (x.ndim)
print (x.shape)
print (x.dtype)


print("Sum is: ", x.sum())
print("Average/Mean value is: ", x.mean())
print("Max value is: ", x.max())
print("Min value is: ", x.min())

#print("Median value is: ", np.median(x))
#print("Correlation coefficient value is: ", np.corrcoef(x))
#print("Standard Deviation is: ", np.std(x))




# Row wise and column wise min
print("Row wise minimum: ", np.amin(x, axis=0))

print("Column wise minimum: ", np.amin(x, axis=1))




#-------------------------------------------------------------------------
"""
Arrays Operations - Basic operations apply element-wise. 
The result is a new array with the resultant elements.
Operations like *= and += will modify the existing array.
"""

import numpy as np
a = np.arange(5) 
print (a)

b = np.arange(5) 
print(b)


x= np.array(list(zip(a,b)))
print (x) 


x = a + b
print (x) 

x = a - b
print (x)

x = a**3
print (x)
 
x = a>3
print (x)

 
x= 10*np.sin(a)
print (x) 

x = a*b
print (x)




# Basics of Matplotlib


import matplotlib.pyplot as plt
import numpy as np

x = [1,2,3,4,5,6,7,8,9,10]
y = [1,2,3,4,5,6,7,8,9,10]
y_sqr = [1,4,9,16,25,36,49,64,81,100]
y_cube =[1,8,27,64,125,216,343,512,729,1000]


      
# Changing the x axes limits of the scale
plt.xlim(0, 10)
# Changing the y axes limits of the scale
plt.ylim(0, 10)
# Or
plt.axis([0, 10, 0, 10]);




# Simple Line plot
# plt.plot(x, y)
# Changing the color of the line
# plt.plot(x, y, color='green') # #000000
# Changing the style of the line
# plt.plot(x, y, linestyle='dashed') # solid dashed  dashdot dotted

# For Plotting Scatter Plot
plt.plot(x, y, 'd', color='black', label="diamond line") # o  .  , x  +  v  ^  <  >  s d 
plt.plot(y,y_sqr , 'o', color="red", label="red label")


# Setting the title
plt.title("A Line Graph")
# Setting the X Label 
plt.xlabel("X")
# Setting the Y Label
plt.ylabel("Y")
# Displaying the legend
plt.legend()


plt.show()


# Showing Different ways of Scatter Plots with plot

rng = np.random.RandomState(0)
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    plt.plot(rng.rand(5), rng.rand(5), marker,
             label="marker='{0}'".format(marker))
plt.legend()
plt.xlim(0, 1.8)



# Scatter Plot with scatter method 

plt.scatter(x, y, marker='.') # o  .  , x  +  v  ^  <  >  s d 

# In this way, the color and size of points can be used to convey information in the visualization, 
# in order to visualize multidimensional data.

rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)

plt.scatter(x, y, c=colors, s=sizes, alpha=0.3)
plt.colorbar();  # show color scale



# 1D Histogram
data = np.random.randn(1000)
plt.hist(data);


plt.hist(data, bins=30, normed=True, alpha=0.5,
         histtype='stepfilled', color='steelblue',
         edgecolor='none');


         
         

# Basics of Statistics for Machine Learning ( Data Exploration )
# Get insights into the data

"""
The major types of Data are:
    Numerical
    Categorical
    Ordinal
"""    

# https://medium.com/technology-nineleaps/basics-of-statistics-for-machine-learning-engineers-bf2887ac716c
    
# Mean, Median, Mode
# (statistics.png)
# Mean
# Mean is given by the total of the values of the samples divided by the number of samples

# x = [10,20,30,40,50]
# mean = (10+20+30+40+50)/5 = 30


# Median
# To calculate the median, sort the values and take the middle value. 
# Now, in case there are even number of values then 
# the average of the two middle values are taken as the median.

# x = [23, 40, 6, 74, 38, 1, 70]
# sorted_x = [1, 6, 23, 38, 40, 70, 74]
# Meadian = 38

# The advantage of the median over the mean is that median is less susceptible to outliers

# So, in situations where there is a high chance that there may be outliers present 
# in the data set, it is wiser to take the median instead of the mean.


# Mode
# Mode represents the most common value in a data set.



# https://medium.com/technology-nineleaps/basics-of-statistics-for-machine-learning-engineers-ii-d25c5a5dac67
# Variance and Standard Deviation


# Variance and Standard Deviation are essentially a measure 
# of the spread of the data in the data set.

# Variance is the average of the squared differences from the mean.
# Calculate the mean
# Calculate the difference from the mean
# find the square of the differences 
# variance is the Sum / Ttotal 

# standard deviation is the square root of the variance

# observations = [23, 40, 6, 74, 38, 1, 70]
# mean = (23+40+6+74+38+1+70) / 7 = 252 /7 = 36
# difference_from_the_mean = [13, -4, 30, -38, -2, 35, -34]
# square_of_the_differences = [169, 16, 900, 1444, 4, 1225, 1156]
# variance = (169+16+900+1444+4+1225+1156)/7 = 4914/7 = 702
# standard deviation = square_root(702)= 26.49

# Standard deviation is an excellent way to identify outliers.
# Data points that lie more than one standard deviation from the mean can be considered unusual. 
# data points that are more than two standard deviations away from the mean are not considered in analysis.





# percentiles and moments

# When a value is given x percentile, 
# this means that x percentage of values in the distribution is below that value


# Moments try to measure the shape of the probability distribution function.
# The zeroth moment is the total probability of the distribution which is 1.

# The first moment is the mean.
# The second moment is the variance
# The third moment is the skew which measures how lopsided the distribution is.
# The fourth moment is kurtosis which is the measure of how sharp is the peak of the graph.

# x = [1,2,6,7]
# first_moment = (1 + 2 + 6 + 7)/ 4 = 16/4 = 4
# sec_moment = (1 + 4 + 36 + 49) / 4 = 90/4 = 22.5
# third_moment = ( 1 + 8 + 216 + 343)/4 = 568/4 = 142



# Covariance and Corelation
# Covariance and Correlation are the tools that we have to measure if the two attributes are related to each other or not.

# Covariance measures how two variables vary in tandem to their means.
# Correlation also measures how two variables move with respect to each other

# A perfect positive correlation means that the correlation coefficient is 1. 
# A perfect negative correlation means that the correlation coefficient is -1. 
# A correlation coefficient of 0 means that the two variables are independent of each other.

# Both correlation and covariance only measure the linear relationship between data.

# Correlation is a special case of covariance when the data is standardized.


#Probability and Statistics

# We use a lot of probability concepts in statistics and hence in machine learning, 
# they are like using the same methodologies. 

#In probability, the model is given and we need to predict the data. 
# While in statistics we start with the data and predict the model. 

#We look at probability and search from data distributions which closely match 
# the data distribution that we have. 

#Then we assume that the function or the model must be the same as the one we 
# looked into in probability theory.


# Conditional Probability and Bayes’ Theorem




"""
Mean, Median, Mode

Let's create some fake income data, centered around 27,000 
with a normal distribution and standard deviation of 15,000, with 10,000 data points. 
Then, compute the mean (average)

"""

import numpy as np

incomes = np.random.normal(27000, 15000, 10000)
print (type(incomes))
print (incomes)
print (len(incomes))
print (incomes.ndim)
print (incomes.shape)
print (incomes.dtype)

print("Mean value is: ", np.mean(incomes))
print("Median value is: ", np.median(incomes))
print("Standard Deviation is: ", np.std(incomes))
print("Correlation coefficient value is: ", np.corrcoef(incomes))



#We can segment the income data into 50 buckets, and plot it as a histogram:
import matplotlib.pyplot as plt
plt.hist(incomes, 20)
plt.show()


#box and whisker plot to show distribution
#https://chartio.com/resources/tutorials/what-is-a-box-plot/
plt.boxplot(incomes)


print("Mean value is: ", np.mean(incomes))
print("Median value is: ", np.median(incomes))

#Adding Bill Gates into the mix. income inequality!(Outliers)
incomes = np.append(incomes, [1000000])

#Median Remains Almost SAME
np.median(incomes)

#Mean Changes distinctly
np.mean(incomes)



#Standard Deviation

import numpy as np
import matplotlib.pyplot as plt

#incomes = np.random.normal(100.0, 50.0, 10000)
incomes = np.random.normal(27000.0, 15000.0, 10000)
plt.hist(incomes, 50)
plt.show()

print (incomes.std())
print (incomes.var())
#The standard deviation is the square root of the variance. 


randNumbers = np.random.randint(5,15,40)
counts = np.bincount(randNumbers)
print (np.argmax(counts))



"""
Code Challenge
  Name: 
    Space Seperated data
  Filename: 
    space_numpy.py
  Problem Statement:
    You are given a 9 space separated numbers. 
    Write a python code to convert it into a 3x3 NumPy array of integers.
  Input:
    6 9 2 3 5 8 1 5 4
  Output:
      [[6 9 2]
      [3 5 8]
      [1 5 4]]
  
  Solutiom:
import numpy as np
num = raw_input("Enter: ").split(" ")
#a.reverse()
num_arr = np.array(num,int)
print np.reshape(num_arr, (3,3))
       
"""




"""
Code Challenge
  Name: 
    E-commerce Data Exploration
  Filename: 
    ecommerce.py
  Problem Statement:
      To create an array of random e-commerce data of total amount spent per transaction. 
      Execute the code snippet below.
          import numpy as np
          import matplotlib.pyplot as plt
          incomes = np.random.normal(100.0, 20.0, 10000)
          print (incomes)
      Segment this incomes data into 50 buckets (number of bars) and plot it as a histogram.
      Find the mean and median of this data using NumPy package.
      Add outliers 
          
  Hint: 
    outlier is an observation that lies an abnormal distance from other values 
    in a random sample from a population) to it to see their effect.
    
"""



"""
Code Challenge
  Name: 
    Normally Distributed Random Data
  Filename: 
    noraml_dist.py
  Problem Statement:
    Create a normally distributed random data with parameters:
    Centered around 150.
    Standard Deviation of 20.
    Total 1000 data points.
    
    Plot the histogram using matplotlib (bucket size =100) and observe the shape.
    Calculate Standard Deviation and Variance. 
"""


"""
Code Challenge
  Name: 
    Random Data
  Filename: 
    random_data.py
  Problem Statement:
    Create a random array of 40 integers from 5 - 15 using NumPy. 
    Find the most frequent value with and without Numpy.

  Solution:
import numpy as np
a = np.random.randint(5,15,40)
d = {}
fin = []

for i in a:
    d[i] = d.get(i,0)+1

for i in d.items():
    j,k = i
    i = k,j
    fin.append(i)

fin.sort()
freq,num = fin[-1]

Or optional Method

num = np.bincount(a).argmax()


print ("The most Frequent Number is",num)
    
"""





"""

# Real World Examples

# Image Feature Extraction
# First, we can map the image into a NumPy array of its pixel values:
# For simplicity’s sake, the image is loaded in grayscale, resulting in a 2d array of 64-bit floats 
# rather than a 3-dimensional MxNx4 RGBA array, with lower values denoting darker spots:
    
from skimage import io
url = ('https://www.history.navy.mil/bin/imageDownload?image=/'
       'content/dam/nhhc/our-collections/photography/images/'
       '80-G-410000/80-G-416362&rendition=cq5dam.thumbnail.319.319.png')
img = io.imread(url, as_grey=True)

fig, ax = plt.subplots()
ax.imshow(img, cmap='gray')
ax.grid(False)

img.shape

img.min(), img.max()

img[0, :10]  # First ten cells of the first row

img[-1, -10:]  # Last ten cells of the last row

img.dtype

#Internally, img is kept in memory as one contiguous block of 648,208 bytes
img.nbytes

#strides is hence a sort of “metadata”-like attribute that tells us how many bytes we need to jump ahead 
#to move to the next position along each axis. 
#We move in blocks of 8 bytes along the rows but need to traverse 8 x 319 = 2,552 bytes to move “down” from one row to another.
 
img.strides









one_d_array = np.array([1,2,3,4,5,6])
print (one_d_array)

# Create a new 2d array
two_d_array = np.array([one_d_array, one_d_array + 6, one_d_array + 12])

# Slice elements starting at row 2, and column 5

two_d_array[1:, 4:]

# Reverse both dimensions (180 degree rotation)

two_d_array[::-1, ::-1]

#Reshaping an Array

np.reshape(a=two_d_array,        # Array to reshape
           newshape=(6,3))       # Dimensions of the new array


#Unravel a multi-dimensional into 1 dimension with np.ravel():

np.ravel(a=two_d_array,
         order='C')         # Use C-style unraveling (by rows)

np.ravel(a=two_d_array,
         order='F')         # Use Fortran-style unraveling (by columns)


#Alternatively, use ndarray.flatten() to flatten a multi-dimensional into 1 dimension and return a copy of the result:
two_d_array.flatten()


#Transpose of the array
two_d_array.T


#Flip an array vertically np.flipud(), upside down :

np.flipud(two_d_array)

  
#Flip an array horizontally with np.fliplr(), left to right:

np.fliplr(two_d_array)


#Rotate an array 90 degrees counter-clockwise with np.rot90():
np.rot90(two_d_array,
         k=2)             # Number of 90 degree rotations

#Shift elements in an array along a given dimension with np.roll():
np.roll(a= two_d_array,
        shift = 1,        # Shift elements 2 positions
        axis = 1)         # In each row

np.roll(a= two_d_array,
        shift = 2,        # Shift elements 2 positions
        axis = 0)         # In each columns

#Join arrays along an axis with np.concatenate():
array_to_join = np.array([[10,20,30],[40,50,60],[70,80,90]])

np.concatenate( (two_d_array,array_to_join),  # Arrays to join
               axis=1)                        # Axis to join upon


# Get the mean of all the elements in an array with np.mean()

np.mean(two_d_array)

# Provide an axis argument to get means across a dimension

np.mean(two_d_array,
        axis = 0)     # Get means of each row

# Get the standard deviation all the elements in an array with np.std()

np.std(two_d_array)


# Provide an axis argument to get standard deviations across a dimension

np.std(two_d_array,
        axis = 0)     # Get stdev for each column

# Sum the elements of an array across an axis with np.sum()

np.sum(two_d_array, 
       axis=1)        # Get the row sums

np.sum(two_d_array,
       axis=0)        # Get the column sums

# Take the square root of each element with np.sqrt()

np.sqrt(two_d_array)


#Take the dot product of two arrays with np.dot(). 
#This function performs an element-wise multiply and then a sum for 1-dimensional arrays (vectors) and matrix multiplication for 2-dimensional arrays.


# Take the vector dot product of row 0 and row 1

np.dot(two_d_array[0,0:],  # Slice row 0
       two_d_array[1,0:])  # Slice row 1
 
#Linear Algebra functions are also available

# Matrices
A = np.mat('1.0 2.0; 3.0 4.0') 
print (A)
print (type(A)) 

print (A.T) # transpose 

X = np.mat('5.0 7.0') 
Y = X.T
print (Y)
 
print (A*Y) # matrix multiplication 
print (A.I) # inverse 

print (np.linalg.solve(A, Y)) # solving linear equation 

#How to check for library version
import numpy as np
print (np.__version__)

"""

