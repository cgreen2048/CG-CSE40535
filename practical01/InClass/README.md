## Milestone 01
Clair Green

#### Can you ignore one or two channels in an HSV color space?
I found that I could ignore the value channel since at any brightness level, 
I wanted to capture the same hue & saturation ranges. To ignore the channel, I tried
large ranges of value to not only capture the same hue & saturation regardless of brightness,
but also not have other objects in frame be captured accidentally as well. I ended up using
the range (100, 255) since that lead to large capture ranges while still not capturing 
extraneous pixels that should not have been captured. This is not possible in RGB since the
brightness/value of color is baked into the RGB channels (not a separate value like in HSV), so 
we cannot have a wide range for one of these channels and expect to capture the same color range.

#### What happens when you present two objects of the same color to the camera?
In my code, I found that the camera would match both objects at two separate points in 
my blue object mask, so I would have to parse through the returned contours and label both of 
them with blue squares as blue objects. They were identified by my camera in two separate
bounding boxes, not one large one covering both.

