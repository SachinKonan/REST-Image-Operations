# Rest Image Operations

Main file contains a Python3.5 Flask server that can easily be created using the free-service, pythonanywhere. Many cloud-based machine learning operations require the manipulation of images either through cropping, edge detection, or general RGB conversion. Lacking a general JS class for doing this, I made a simple Web server which can be accessed through get requests, as shown in the html example, and simply requires input data. For example, when drawing rectangles, the user must pass in the rectangle's top-left coordinates, width, and height in a JSON.  

Additionally has a developing eye-detector, which uses a simple MLP model which has been pre-trained locally for prediction. 

*Note: The Flask app requires the download of Numpy, Scipy, and Matplotlib, but those are provided with the creation of a pythonanywhere free server. 

![alt text](https://github.com/SachinKonan/REST-Image-Operations/blob/master/harden_rect.jpg "James Harden after cvtRect")

