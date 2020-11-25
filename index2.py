from swap import SWAP

img2 = 'images/bradley_cooper.jpg'
# img2 = 'images/bradley_cooper_.jpg'
img1 = 'images/jim_carrey.jpg'

SWAP.enable_visualization()
swap = SWAP(img1, img2)

swap.swap()

# print(swap.img1)
# swap.first_face_landmarks()
# swap.second_face_landmarks()

# swap.test()

# swap.first_convex_hull()
# swap.second_convex_hull()
#
# swap.delaunay_test()

# swap.show_images()
