#!/usr/bin/env python2.7

# see README for description of the lab.

import sys

from PIL import Image as PIL_Image

RED = (255, 0, 0)

# set debug to False before turning in codedebug = True
debug = False

class Image:
    # Create an image, either from a file with a given filename,
    # or a blank image with a given size = (width, height).
    def __init__(self, filename = None, size = None):
        assert not (filename and size)
        if filename:
            img = PIL_Image.open(filename)
            assert img.mode == "RGB", "unexpected mode: " + img.mode
            raw_data = img.getdata()
            self.width, self.height = img.size
            self.data = [raw_data[i] for i in range(self.width * self.height)]
        else:
            self.width, self.height = size
            assert 0 < self.width and 0 < self.height
            self.data = [(0,0,0)] * (self.width*self.height)

    # Each pixel in the image is a triple (r,g,b) of numbers.

    # get the (r,g,b) triple at a given row and column
    def get_pixel(self, row, col):
        assert 0 <= row < self.height and 0 <= col < self.width
        return self.data[row*self.width + col]

    # set the (r,g,b) triple at a given row and column
    def set_pixel(self, row, col, rgb):
        assert 0 <= row < self.height and 0 <= col < self.width
        assert len(rgb) == 3
        self.data[row*self.width + col] = rgb

    # save image to file named filename
    def save(self, filename):
        image = PIL_Image.new("RGB", (self.width, self.height))
        image.putdata(self.data)
        image.save(filename)

    def _distance(self, rgb1, rgb2):
        return sum((a-b)**2 for a,b in zip(rgb1, rgb2))

    # return the "energy" of a given pixel in the image.
    def energy(self, row, col):
        result = 0
        pixel = self.get_pixel(row, col)
        for c in (col-1, col+1):
            if 0 <= c < self.width:
                result += self._distance(pixel, self.get_pixel(row, c))
        for r in (row-1, row+1):
            if 0 <= r < self.height:
                result += self._distance(pixel, self.get_pixel(r, col))
        if col in (0, self.width): result *= 2
        return result

# The next function is only for demonstrating the use of the Image class.
# Given an image file, it creates a new file with the image flipped left-to-right.
def flip(filename, new_filename):
    original = Image(filename)
    width, height = original.width, original.height

    flipped = Image(size = (width, height))
    for row in range(height):
        for col in range(width):
            rgb = original.get_pixel(row, col)
            flipped.set_pixel(row, width-col-1, rgb)

    flipped.save(new_filename)

# flip("inputs/tower.jpg", "tower-flipped.jpg")

# Utility function - argmin.
# Return the i in indices minimizing function(i).

def argmin(function, indices): return min(indices, key = function)

# For example, argmin(lambda i: 1 + i**2, [-1,0,1,2]) will return 0.
# Note that (lambda i: 1+i**2) builds an "anonymous" function that, given i, returns 1+i**2.

# This is the main routine.
# Given an image, find a minimum-energy seam from the top row to the bottom row.
def find_min_energy_seam(image):
    width, height = image.width, image.height

    assert width > 0

    # Compute ME[i][j], be min energy of any seam
    # running from anywhere in the top row down to pixel i,j.
    # Keep track of the corresponding seams by storing
    # the parent of the best seam ending at (i,j) in parents[i][j].

    # Initialize ME and parents as 2-dimensional width x height arrays.
    ME = [ [None] * width for i in range(height) ]
    parents = [ [None] * width for i in range(height) ]

    # Initialize the first row using the boundary condition of the recurrence.
    for j in range(width): ME[0][j] = image.energy(0, j)
  
    for i in range(height):
        for j in range(width):
            if j == 0:
                ME[i][j] = image.energy(i,j) + min( ME[i-1][j], ME[i-1][j+1])
                parents[i][j] = argcmin(lambda A: ME[i][j], (j), (j-1))  
            elif j == width -1:
                ME[i][j] = image.energy(i,j) + min( ME[i-1][j-1], ME[i-1][j])    
                parents[i][j] = argcmin(lambda A: ME[i][j], (j-1), (j))  
            else: 
                ME[i][j] = image.energy(i,j) + min( ME[i-1][j-1], ME[i-1][j], ME[i-1][j+1])    
                parents[i][j] = argcmin(lambda A: ME[i][j], (j-1), (j), (j-1))  

 # Reconstruct and return the minimum energy seam from top to bottom.
    best_seam = [None]*height
    best_seam[height-1] = argmin(lambda j: ME[height-1][j], range(width))

    # make sure to leave this line in your turned-in code.
    print "seam energy", ME[height-1][best_seam[height-1]]
    
    for i in range(height-1,0,-1):
        best_seam[i-1] = parents[i][best_seam[i]]

    assert all(0 <= best_seam[i] < width for i in range(height))

    # The returned seam is encoded in a 1-dimensional array
    # with an entry for each row giving the index of the column
    # that the seam passes through in that row.
    return best_seam

# Given an image and a seam (where seam[i] is the index of the seam in row i),
# return a new image with the seam removed.
# (In each row, all pixels to the right of the seam are shifted left one pixel.)

def carve_seam_from_image(image, seam):
    width, height = image.width, image.height
    assert all(0 <= seam[i] < width for i in range(height))
    assert width > 1
    carved = Image(size = (width - 1, height))
    for i in range(height):
        to_delete = seam[i]
        for j in range(to_delete):
            rgb = image.get_pixel(i, j)
            carved.set_pixel(i, j, rgb)
        for j in range(to_delete+1, width):
            rgb = image.get_pixel(i, j)
            carved.set_pixel(i, j-1, rgb)
    return carved

# Given an image and a seam (where seam[i] is the index of the seam in row i),
# return a new image highlighting the seam in the original image.
# This is useful mainly for debugging, to sanity check your seams.

def show_seam(image, seam):
    width, height = image.width, image.height

    assert all(0 <= seam[i] < width for i in range(height))

    highlighted = Image(size = (width, height))
    for i in range(0, height):
        for j in range(0, width):
            rgb = image.get_pixel(i, j)
            highlighted.set_pixel(i, j, rgb)
        highlighted.set_pixel(i, seam[i], RED)
    return highlighted
    
## main code begins here.

# invoke with two arguments: carve tower.jpg 2
# This will produce an output file with two seams removed,
# reducing the width by 2.  The output filename is determined
# from the filename by inserting the number of iterations
# and a tag.  The first arg defaults to inputs/tower.jpg.
# The second argument defaults to 1.

filename = sys.argv[1] if len(sys.argv) >= 2 else "inputs/tower.jpg"
iterations = int(sys.argv[2]) if len(sys.argv) >= 3 else 1

def output_filename(filename, tag, iteration):
    return filename.replace(".", "-" + str(iteration) + "-" + tag + ".")

image = Image(filename)
for i in range(iterations):
    seam = find_min_energy_seam(image)

    carved = carve_seam_from_image(image, seam)

    if debug:
        highlighted = show_seam(image, seam)
        highlighted.save(output_filename(filename, "seam", i))
        carved.save(output_filename(filename, "carved", i))

    image = carved

image.save(output_filename(filename, "carved", i))
