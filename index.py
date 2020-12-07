from utils.swap_test import SWAP_T

# img2 = 'images/dadario.jpg'
# img1 = 'images/emma.jpg'

img1 = 'images/bradley_cooper.jpg'
img2 = 'images/jim_carrey.jpg'

SWAP_T.enable_visualization()
swap = SWAP_T(img1, img2)

SWAP_T.enable_visualization()
swap.first_face_landmarks()

SWAP_T.enable_visualization()
swap.second_face_landmarks()

SWAP_T.enable_visualization(False)
swap.first_convex_hull()

SWAP_T.enable_visualization(False)
swap.second_convex_hull()

SWAP_T.enable_visualization(False)
swap.delaunay_triangle_1()

# SWAP_T.enable_visualization(False)
# swap.delaunay_triangle_2()
#
# SWAP_T.enable_visualization(False)
# swap.combine_triangle()
#
# SWAP_T.enable_visualization(False)
# swap.final_show()

swap.show_images()
