from swap import SWAP
import dlib

img2 = 'images/jim_carrey.jpg'
img1 = 'images/bradley_cooper.jpg'

# img1 = 'images/bradley_cooper.jpg'
# img2 = 'images/jim_carrey.jpg'

SWAP.enable_visualization()
swap = SWAP(img1, img2)

SWAP.enable_visualization(False)
swap.first_face_landmarks()

SWAP.enable_visualization(False)
swap.first_convex_hull()

SWAP.enable_visualization(False)
swap.delaunay_triangle_1()

SWAP.enable_visualization(False)
swap.second_face_landmarks()

SWAP.enable_visualization(False)
swap.delaunay_triangle_2()

SWAP.enable_visualization(False)
swap.combine_triangle()

SWAP.enable_visualization()
swap.after_math()

swap.show_images()
