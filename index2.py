from utils.swap import SWAP

# img1 = 'images/bradley_cooper.jpg'
# img2 = 'images/jim_carrey.jpg'

img1 = 'images/chris-hems.jpg'
img2 = 'images/tom-2.jpg'

SWAP.enable_visualization()
swap = SWAP(img1, img2)

swap.swap()

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
