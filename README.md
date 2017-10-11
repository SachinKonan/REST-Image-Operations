# Rest Image Operations

Main file contains a Python3.5 Flask server that can easily be created using the free-service, pythonanywhere. Many cloud-based machine learning operations require the manipulation of images either through cropping, edge detection, or general RGB conversion. Lacking a general JS class for doing this, I made a simple Web server which can be accessed through get requests, as shown in the html example, and simply requires input data. For example, when drawing rectangles, the user must pass in the rectangle's top-left coordinates, width, and height in a JSON.  
