from utils.swap import SWAP

# img1 = 'images/bradley_cooper.jpg'
# img2 = 'images/jim_carrey.jpg'

img2 = 'images/lawerence.jpg'
# img2 = 'images/tom-2.jpg'
img1 = 'images/jolie-lawrence.jpg'

SWAP.enable_visualization()
swap = SWAP(img1, img2)

swap.my()

# swap.test()

# print(swap.img1)
# swap.first_face_landmarks()
# swap.second_face_landmarks()

# swap.test()

# swap.first_convex_hull()
# swap.second_convex_hull()
#
# swap.delaunay_test()

# swap.show_images()
